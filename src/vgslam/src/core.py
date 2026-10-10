import numpy as np
import small_gicp

from .helpers import *
from .models import *
from .scandb import ScanDB
from .map_builder import MapBuilder

from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se2 import PoseSE2
from graphslam.graph import Graph

from typing import Literal

# save&read
import pickle
import json, struct

class VGSLAM:
    def __init__(
        self,
        # mapping
        map_resolution: float = 0.05,
        map_max_cells: int = 16_000_000,
        map_occ_dec: float = 0.85,
        map_free_inc: float = 0.4,
        map_l_min: float = -4.0,
        map_l_max: float = 4.0,
        map_initial_size: int = 128,
        map_pad_cells: int = 32,
        
        # graph optimization
        dxy_var: float = 0.2, 
        dtheta_var: float = np.deg2rad(3), 
        
        # loop closure
        min_chain_size: int = 5, 
        loop_closure_max_dist: float = 0.45,
        loop_closure_max_ang: float = np.deg2rad(40),
        loop_closure_min_correction: float = 0.04,
        loop_closure_min_angle_correction: float = np.deg2rad(3),
        loop_accept_fitness: float = 0.7,   
        coarse_search_max_candidates: int = 5,
        
        # vertex creation and sub-vertex handling
        min_distance_to_vertex: float = 0.4,
        min_angle_to_vertex: float = np.deg2rad(10),
        min_distance_to_subvertex: float = 0.08,
        min_angle_to_subvertex: float = np.deg2rad(5),
        max_scans_no_subvertex: float = 7,
        vertex_accept_fitness: float = 0.6, 
        vertex_rmse_tolerance: float = 0.2,
        
        # keyframe matching
        keyframe_match_distance_tolerance: float = 0.15,
        keyframe_match_angle_tolerance: float = float(np.deg2rad(5)),
        noise_trans_threshold: float = 0.005,
        noise_rot_threshold: float = np.deg2rad(0.2),
        
        # icp
        matcher_max_correspondence_distance: float = 0.5,
        matcher_mode: Literal['ICP', 'GICP', 'PLANE_ICP'] = "GICP",
        
        # debugging log
        verbose: bool = False
    ) -> None:
        # mapping
        self.map_builder = MapBuilder(
            map_resolution,
            map_max_cells,
            map_occ_dec,
            map_free_inc,
            map_l_min,
            map_l_max,
            map_initial_size,
            map_pad_cells
        )
        self._pending_scans: list[PositionedCloud] = []
        
        # graph
        self.next_vertex_id = 0
        self.vertices: dict[int, Vertex] = {}
        self.edges: list[EdgeOdometry] = []
        self._graph = Graph([], [])

        # scan
        self._current_global_pose = np.eye(4)
        self._last_vertex_scan = None
        self._prev_scan = None
        self._saved_scans: dict[int, PositionedCloud] = {}
        self._last_subkeyframe = None
        self._scans_no_correction = 0
        self._scandb = ScanDB()
        
        # variables
        self._min_distance_to_vertex = min_distance_to_vertex
        self._min_angle_to_vertex = min_angle_to_vertex
        self._min_distance_to_subvertex = min_distance_to_subvertex
        self._min_angle_to_subvertex = min_angle_to_subvertex
        self._min_chain_size = min_chain_size
        self._loop_closure_max_dist = loop_closure_max_dist
        self._loop_closure_max_ang = loop_closure_max_ang
        self._loop_closure_min_correction = loop_closure_min_correction
        self._loop_closure_min_angle_correction = loop_closure_min_angle_correction
        self._vertex_acc_fitness = vertex_accept_fitness
        self._loop_acc_fitness = loop_accept_fitness
        self._vertex_rmse_tol = vertex_rmse_tolerance
        self._vert_match_dist_tol = keyframe_match_distance_tolerance
        self._vert_match_ang_tol = keyframe_match_angle_tolerance
        self._dxy_var = dxy_var
        self._dtheta_var = dtheta_var
        self._matcher_max_corr_distance = matcher_max_correspondence_distance
        self._max_scans_no_correction = max_scans_no_subvertex
        self._coarse_search_max_candidates = coarse_search_max_candidates
        self._noise_trans_threshold = noise_trans_threshold
        self._noise_rot_threshold = noise_rot_threshold
        self._matcher_mode = matcher_mode
        
        # debug
        self._verbose = verbose
    
    def process_scan(self, scan: PositionedCloud) -> tuple[np.ndarray | None, bool]:
        """
        Process new lidar scan with odometry
        """
        
        pos, closed = self._process_scan(scan)
        
        if closed:
            self._pending_scans.clear()
            self.map_builder.rebuild(self._saved_scans.values())
            
            print("clear and rebuild")
        else:
            if scan.estimated_pos is not None:
                self._pending_scans.append(scan)
            
            if len(self._pending_scans) >= 8:
                self.map_builder.add_scans(self._pending_scans)
                self._pending_scans.clear()

        return pos, closed
    
    def _process_scan(self, scan: PositionedCloud) -> tuple[np.ndarray | None, bool]:
        # first call - initialize
        if self._prev_scan is None or self._last_vertex_scan is None or self._last_subkeyframe is None:
            scan.estimated_pos = scan.pos
            
            self._prev_scan = scan
            self._last_vertex_scan = scan
            self._last_subkeyframe = scan
            
            pose_id = self._add_odom_pose(scan.pos, None)
            self._saved_scans[pose_id] = scan
        
            return None, False
        
        # correct using ICP if moved too much from last sub-keyframe or scans correction expired
        from_skf = np.linalg.norm(relative_pose(self._last_subkeyframe.pos, scan.pos)[:2])
        if from_skf > self._min_distance_to_subvertex or self._scans_no_correction >= self._max_scans_no_correction:
            self._scans_no_correction = 0
            
            # estimate transform from previous sub-keyframe to current
            T_sub_current, (fitness, rmse) = self._process_scans(
                scan,
                self._last_subkeyframe, 
                relative_pose_4x4(self._last_subkeyframe.pos, scan.pos)
            )
            
            # if match is bad - fallback to odometry integration
            # so no movement is lost
            if fitness <= self._vertex_acc_fitness or rmse >= self._vertex_rmse_tol:
                # scan.estimated_pos = transform_4x4_to_2d_pose(self._current_global_pose)
                # self._last_subkeyframe = scan
                return self._proc_fallback_odom(scan)
            
            # if previous sub-keyframe has no estimated position
            # fallback to odometry (should never happen)
            if self._last_subkeyframe.estimated_pos is None:
                return self._proc_fallback_odom(scan)
            
            # update global position using new transform            
            self._current_global_pose = pose2d_to_transform(self._last_subkeyframe.estimated_pos) @ T_sub_current
            
            # update positions and cached scans
            pose_2d = transform_4x4_to_2d_pose(self._current_global_pose)
            scan.estimated_pos = pose_2d
            self._last_subkeyframe = scan
            self._prev_scan = scan
            
            closed = False
            
            # should never happen
            if self._last_vertex_scan.estimated_pos is None:
                return pose_2d, False

            # metrics for adding a new vertex
            distance = np.linalg.norm((pose_2d - self._last_vertex_scan.estimated_pos)[:2])
            dtheta = pose_2d[2] - self._last_vertex_scan.estimated_pos[2]
            dtheta = abs(normalize_angle(dtheta))
            
            # add vertex only if distance or angle change is bigger than threshold 
            if distance >= self._min_distance_to_vertex or dtheta >= self._min_angle_to_vertex:
                # get relative from accurate previously estimated poses
                rel_over = relative_pose_4x4(self._last_vertex_scan.estimated_pos, scan.estimated_pos)
                T_prevVertex_current, (fitness, rmse) = self._process_scans(
                    scan, 
                    self._last_vertex_scan, 
                    rel_override=rel_over
                )
                
                if self._verbose:
                    print(
                        f"ICP fitness={fitness:.3f}, "
                        f"RMSE={rmse:.3f}, "
                        f"translation={...}, "
                        f"rotation={...}"
                    )
                
                # if match is too bad to be true - skip vertex placement
                if fitness < 0.25:
                    return pose_2d, False

                # by default we use transform between estimated poses
                tr = rel_pose = relative_pose(self._last_vertex_scan.estimated_pos, scan.estimated_pos)
                
                # ...but if match is really good - we can use estimated delta to be more accurate
                if fitness >= self._vertex_acc_fitness and rmse < self._vertex_rmse_tol:
                    transf_2d = transform_4x4_to_2d_pose(T_prevVertex_current)
                    
                    transf_err = np.linalg.norm((transf_2d[:2] - rel_pose[:2]))
                    transf_err_rel = transf_err / max(np.linalg.norm(rel_pose[:2]), 0.1)
                    rot_err = abs(normalize_angle(transf_2d[2] - rel_pose[2]))
                    if transf_err_rel < self._vert_match_dist_tol and rot_err < self._vert_match_ang_tol:
                        tr = transf_2d

                # add vertex and register scan
                pose_id = self._add_odom_pose(
                    pose_2d,
                    tr
                )
                self._scandb.add_vertex_scan(
                    pose_id,
                    scan,
                )
                
                self._saved_scans[pose_id] = scan
                self._last_vertex_scan = scan

                closed = self.try_to_close_loop()
            
            return pose_2d, closed
        
        else:
            # just increment position by odometry
            return self._proc_fallback_odom(scan)
            
    def _proc_fallback_odom(self, scan: PositionedCloud):
        if self._prev_scan is None:
            return None, False

        d = relative_pose(self._prev_scan.pos, scan.pos)
        trans = np.linalg.norm(d[:2])
        rot = abs(d[2])
        
        if trans < self._noise_trans_threshold and rot < self._noise_rot_threshold:
            scan.estimated_pos = self._prev_scan.estimated_pos
            self._prev_scan = scan
            return scan.estimated_pos, False

        self._scans_no_correction += 1

        T_prev_current = relative_pose_4x4(
            self._prev_scan.pos,
            scan.pos
        )

        self._current_global_pose = (
            self._current_global_pose @ T_prev_current
        )

        scan.estimated_pos = transform_4x4_to_2d_pose(
            self._current_global_pose
        )

        self._prev_scan = scan

        return scan.estimated_pos, False
        
    def _process_scans(self, source: PositionedCloud, target: PositionedCloud, rel_override: np.ndarray | None = None):
        # grab cached or newly created scan data
        pcd_src = source.gicp()
        pcd_tgt = target.gicp()
        tree = target.kdtree()
        
        result = small_gicp.align(
            pcd_tgt, pcd_src, tree,
            relative_pose_4x4(target.pos, source.pos) if rel_override is None else rel_override,
            registration_type=self._matcher_mode,
            max_correspondence_distance=self._matcher_max_corr_distance,
            verbose=self._verbose
        )
        
        T_target_source = result.T_target_source

        # calculate transform accuracy metrics
        fitness, inlier_rmse = evaluate_registration(
            source.cloud,
            target.kdtree(),
            T_target_source,
            max_correspondence_distance=self._matcher_max_corr_distance
        )
        
        return T_target_source, (fitness, inlier_rmse)
    
    def _find_best_match(self, 
        sources: list[PositionedCloud], 
        target: PositionedCloud, 
        min_confidence = 0.9, 
        rmse_max = 0.1
    ) -> tuple[int, np.ndarray | None, tuple]:
        best_score = -float('inf')
        best = (-1, None, (0, 0))

        for i, source in enumerate(sources):
            T_new, (fitness, rmse) = self._process_scans(source, target)

            if fitness < min_confidence or rmse > rmse_max:
                continue
            
            score = fitness - 0.5 * rmse

            if best is None or score > best_score:
                best_score = score
                best = (i, T_new, (fitness, rmse))
        
        i, T, stats = best
        if i > -1:
            return (i, transform_4x4_to_2d_pose(T), stats)
    
        else:
            return best
    
    def _add_odom_pose(self, est_pos: np.ndarray, dpos: np.ndarray | None):
        # add a new vertex
        self.vertices[self.next_vertex_id] = Vertex(
            vertex_id=self.next_vertex_id,
            pose=PoseSE2(
                position=est_pos[:2],
                orientation=est_pos[2]
            ),
            fixed=self.next_vertex_id == 0
        )
        
        # and join it if dpos is provided
        if self.next_vertex_id > 0 and dpos is not None:
            self._add_edge(
                self.next_vertex_id - 1,
                self.next_vertex_id,
                dpos[:2],
                dpos[2],
                1.0
            )
        
        self.next_vertex_id += 1
        return self.next_vertex_id - 1
    
    def _add_edge(self, id_a: int, id_b: int, odom_dpos: np.ndarray, odom_dtheta: float, var_mul: float):
        # add an edge with variances
        self.edges.append(EdgeOdometry(vertex_ids=[id_a, id_b], information=np.array([
            [1 / (self._dxy_var * var_mul), 0, 0],
            [0, 1 / (self._dxy_var * var_mul), 0],
            [0, 0, 1 / (self._dtheta_var * var_mul)],
        ]), estimate=PoseSE2(
            position=odom_dpos,
            orientation=odom_dtheta
        )))

    def try_to_close_loop(self) -> bool:
        if self._prev_scan is None or self._prev_scan.estimated_pos is None:
            return False

        if self._verbose:
            print("trying to close loop")

        # ids are always in distance-ascending order
        coarse_ids, _ = self._scandb.search_candidates(self._prev_scan, self._coarse_search_max_candidates, self._min_chain_size)
        blacklist = get_topological_neighbors(self.next_vertex_id - 1, self.edges, self._min_chain_size)

        vertices = [self.vertices[id] for id in coarse_ids if id not in blacklist]

        # get the best match
        scans = [self._saved_scans[v.id] for v in vertices]
        idx, pose, (fitness, rmse) = self._find_best_match(scans, self._prev_scan, self._loop_acc_fitness, self._vertex_rmse_tol)

        # if match found - calculate displacement error and 
        # close the loop if it is acceptable
        if pose is not None and idx >= 0:
            predicted = relative_from_vertices(self.vertices[self.next_vertex_id - 1], vertices[idx])

            err = np.linalg.norm((pose - predicted)[:2])

            angle_err = pose[2] - predicted[2]
            angle_err = abs(normalize_angle(angle_err))

            if self._verbose:
                print("pose_err angle_err" , err, angle_err)
                print("predicted_pose estimated_pose", predicted, pose)

            if err <= self._loop_closure_max_dist and angle_err <= self._loop_closure_max_ang and (
                err > self._loop_closure_min_correction or
                angle_err > self._loop_closure_min_angle_correction
            ):
                if self._verbose:
                    print("loop closure stats")
                    print("pose_loop pose_prev", vertices[idx].pose.position, self.vertices[self.next_vertex_id - 1].pose.position)
                    print("fitness:", fitness, "rmse:", rmse)

                self._add_edge(self.next_vertex_id - 1, vertices[idx].id, pose[:2], pose[2], 0.5)

                if self._verbose:
                    print("added edge [", vertices[idx].id, self.next_vertex_id - 1, "]")

                self.optimize()

                return True

        if self._verbose:
            print("closure failed")

        return False

    def create_occupancy_grid(self, occupied_thresh: float = 0.4) -> OccupancyGrid:
        if len(self._pending_scans) > 0:
            self.map_builder.add_scans(self._pending_scans)
            self._pending_scans.clear()

        return self.map_builder.to_occupancy_grid(occupied_thresh)

    def optimize(self):
        self._graph._vertices = list(self.vertices.values())
        self._graph._edges = self.edges
        self._graph._initialize()

        res = self._graph.optimize(verbose=self._verbose)

        for vertex in self.vertices.values():
            self._saved_scans[vertex.id].estimated_pos = vertex.pose

        last_vertex = self.vertices[self.next_vertex_id - 1]
        self._current_global_pose = pose2d_to_transform(
            last_vertex.pose
        )

        if self._verbose:
            print("optimized graph in", res.duration_s)

    def save_to_file(self, fp: str):
        if self._pending_scans:
            self.map_builder.add_scans(self._pending_scans)
            self._pending_scans.clear()
        
        vertices = self.vertices
        edges = self.edges
        vertices = [
            [int(k), *map(float, v.pose)] 
            for k, v in vertices.items()
        ]
        edges = [
            [
                list(map(int, v.vertex_ids)),
                [*map(float, v.estimate)],
                v.information.flatten().astype(float).tolist(),
            ]
            for v in edges
        ]
        
        parameters = {
            # --- mapping ---
            "map_resolution": self.map_builder.resolution,
            "map_max_cells": self.map_builder._max_cells,
            "map_occ_dec": self.map_builder._occ_dec,
            "map_free_inc": self.map_builder._free_inc,
            "map_l_min": self.map_builder._l_min,
            "map_l_max": self.map_builder._l_max,
            "map_initial_size": self.map_builder._initial_size,
            "map_pad_cells": self.map_builder._pad_cells,

            # --- graph optimization ---
            "dxy_var": self._dxy_var,
            "dtheta_var": self._dtheta_var,

            # --- loop closure ---
            "min_chain_size": self._min_chain_size,
            "loop_closure_max_dist": self._loop_closure_max_dist,
            "loop_closure_max_ang": self._loop_closure_max_ang,
            "loop_closure_min_correction": self._loop_closure_min_correction,
            "loop_closure_min_angle_correction": self._loop_closure_min_angle_correction,
            "loop_accept_fitness": self._loop_acc_fitness,
            "coarse_search_max_candidates": self._coarse_search_max_candidates,

            # --- vertex creation ---
            "min_distance_to_vertex": self._min_distance_to_vertex,
            "min_angle_to_vertex": self._min_angle_to_vertex,
            "min_distance_to_subvertex": self._min_distance_to_subvertex,
            "min_angle_to_subvertex": self._min_angle_to_subvertex,
            "max_scans_no_subvertex": self._max_scans_no_correction,
            "vertex_accept_fitness": self._vertex_acc_fitness,
            "vertex_rmse_tolerance": self._vertex_rmse_tol,

            # --- matching ---
            "keyframe_match_distance_tolerance": self._vert_match_dist_tol,
            "keyframe_match_angle_tolerance": self._vert_match_ang_tol,
            "noise_trans_threshold": self._noise_trans_threshold,
            "noise_rot_threshold": self._noise_rot_threshold,

            # --- icp ---
            "matcher_max_correspondence_distance": self._matcher_max_corr_distance,
            "matcher_mode": self._matcher_mode,

            # --- runtime state ---
            "next_vertex_id": int(self.next_vertex_id),

            # --- graph ---
            "graph_vertices": vertices,
            "graph_edges": edges,
        }
        
        params_encoded = json.dumps(parameters, separators=(',', ':')).encode("utf-8")
        
        scandb_encoded = self._scandb.to_bytes()
        
        scans = {int(k): v.to_bytes() for k, v in self._saved_scans.items()}
        scans_encoded = pickle.dumps(scans)
        
        with open(fp, "wb") as f:
            for blob in (params_encoded, scandb_encoded, scans_encoded):
                f.write(struct.pack(">I", len(blob)))
                f.write(blob)

    @staticmethod
    def load_from_file(fp: str) -> "VGSLAM":
        with open(fp, "rb") as f:
            def read_blob():
                n = struct.unpack(">I", f.read(4))[0]
                return f.read(n)

            params_encoded = read_blob()
            scandb_encoded = read_blob()
            scans_encoded = read_blob()

        parameters = json.loads(params_encoded.decode("utf-8"))
        scans_raw = pickle.loads(scans_encoded)
        
        slam = VGSLAM(
            map_resolution=parameters["map_resolution"],
            map_max_cells=parameters["map_max_cells"],
            map_occ_dec=parameters["map_occ_dec"],
            map_free_inc=parameters["map_free_inc"],
            map_l_min=parameters["map_l_min"],
            map_l_max=parameters["map_l_max"],
            map_initial_size=parameters["map_initial_size"],
            map_pad_cells=parameters["map_pad_cells"],

            dxy_var=parameters["dxy_var"],
            dtheta_var=parameters["dtheta_var"],

            min_chain_size=parameters["min_chain_size"],
            loop_closure_max_dist=parameters["loop_closure_max_dist"],
            loop_closure_max_ang=parameters["loop_closure_max_ang"],
            loop_closure_min_correction=parameters["loop_closure_min_correction"],
            loop_closure_min_angle_correction=parameters["loop_closure_min_angle_correction"],
            loop_accept_fitness=parameters["loop_accept_fitness"],
            coarse_search_max_candidates=parameters["coarse_search_max_candidates"],

            min_distance_to_vertex=parameters["min_distance_to_vertex"],
            min_angle_to_vertex=parameters["min_angle_to_vertex"],
            min_distance_to_subvertex=parameters["min_distance_to_subvertex"],
            min_angle_to_subvertex=parameters["min_angle_to_subvertex"],
            max_scans_no_subvertex=parameters["max_scans_no_subvertex"],
            vertex_accept_fitness=parameters["vertex_accept_fitness"],
            vertex_rmse_tolerance=parameters["vertex_rmse_tolerance"],

            keyframe_match_distance_tolerance=parameters["keyframe_match_distance_tolerance"],
            keyframe_match_angle_tolerance=parameters["keyframe_match_angle_tolerance"],
            noise_trans_threshold=parameters["noise_trans_threshold"],
            noise_rot_threshold=parameters["noise_rot_threshold"],

            matcher_max_correspondence_distance=parameters["matcher_max_correspondence_distance"],
            matcher_mode=parameters["matcher_mode"],
        )
        
        for v_data in parameters["graph_vertices"]:
            vid = int(v_data[0])
            x, y, theta = float(v_data[1]), float(v_data[2]), float(v_data[3])
            slam.vertices[vid] = Vertex(
                vertex_id=vid,
                pose=PoseSE2(position=np.array([x, y]), orientation=theta),
                fixed=(vid == 0),
            )
        
        for e_data in parameters["graph_edges"]:
            ids = list(map(int, e_data[0]))
            x, y, theta = float(e_data[1][0]), float(e_data[1][1]), float(e_data[1][2])
            info = np.asarray(e_data[2], dtype=np.float64).reshape(3, 3)

            slam.edges.append(EdgeOdometry(
                vertex_ids=ids,
                information=info,
                estimate=PoseSE2(position=np.array([x, y]), orientation=theta),
            ))
        
        slam.next_vertex_id = int(parameters["next_vertex_id"])
        
        slam._scandb = ScanDB.from_bytes(scandb_encoded)
        slam._saved_scans = {
            int(k): PositionedCloud.from_bytes(v)
            for k, v in scans_raw.items()
        }
        
        if slam.next_vertex_id > 0:
            last_id = slam.next_vertex_id - 1
            last_scan = slam._saved_scans[last_id]

            slam._last_vertex_scan = last_scan
            slam._prev_scan = last_scan
            slam._last_subkeyframe = last_scan

            p = last_scan.estimated_pos
            slam._current_global_pose = pose2d_to_transform(p)

        slam.map_builder.rebuild(slam._saved_scans.values())
        
        return slam
