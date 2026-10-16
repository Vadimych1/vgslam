import numpy as np

from .helpers import *
from .models import *
from .scandb import ScanDB
from .map_builder import MapBuilder
from .graph import PoseGraph
from .icp_matcher import ICPMatcher, ICP_MATCHER_MODE
from .hypotetical_localization import LocalizationHypothesisSelection, LocalizationHypothesis

# save&read
import pickle
import json, struct

from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se2 import PoseSE2

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
        loop_closure_edge_uncertainty: float = 0.5,
        odometry_edge_uncertainty: float = 1.0,
        
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
        matcher_mode: ICP_MATCHER_MODE = "GICP",
        
        # probabilistic location
        use_probabilistic_loc: bool = True,
        loc_max_tries: int = 16,
        loc_max_hypotheses: int = 8,
        loc_init_min_fitness: float = 0.3,
        loc_commit_prob: float = 0.85,          # absolute posterior for the winner
        loc_commit_margin: float = 3.0,         # log-odds vs. runner-up
        loc_min_scans: int = 3,                 # evidence before we commit
        loc_rmse_sigma: float = 0.05,           # for ICP likelihood
        loc_odom_sigma: float = 0.3,            # m, for odometry consistency
        loc_odom_ang_sigma: float = float(np.deg2rad(10)),
        loc_prune_thresh: float = 1e-3,
        loc_dup_dist: float = 0.5,              # m, dedup threshold
        loc_dup_ang: float = float(np.deg2rad(30)),
        
        # debugging log
        verbose: bool = False
    ):
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
        self._loop_closure_edge_uncertainty = loop_closure_edge_uncertainty
        self._odometry_edge_uncertainty = odometry_edge_uncertainty
        self._graph = PoseGraph(
            dxy_var,
            dtheta_var,
            loop_closure_edge_uncertainty,
            odometry_edge_uncertainty,
        )

        # scan
        self._current_global_pose = np.eye(4)
        self._last_vertex_scan = None
        self._prev_scan = None
        self._saved_scans: dict[int, PositionedCloud] = {}
        self._last_subkeyframe = None
        self._scans_no_correction = 0
        self._scandb = ScanDB()
        self._matcher = ICPMatcher(
            matcher_max_correspondence_distance,
            matcher_mode,
        )
        
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
        self._matcher_max_corr_distance = matcher_max_correspondence_distance
        self._max_scans_no_correction = max_scans_no_subvertex
        self._coarse_search_max_candidates = coarse_search_max_candidates
        self._noise_trans_threshold = noise_trans_threshold
        self._noise_rot_threshold = noise_rot_threshold
        self._matcher_mode = matcher_mode
        
        # localization hypothesis
        self._prob_loc_done = not use_probabilistic_loc
        self._use_prob_loc = use_probabilistic_loc
        self._loc_tries = 0
        self._loc_max_tries = loc_max_tries
        if use_probabilistic_loc:
            self._hypothesis_localizer = LocalizationHypothesisSelection(
                self._saved_scans,
                self._matcher,
                self._graph,
                self._scandb,
                loc_max_hypotheses,
                loc_init_min_fitness,
                loc_commit_prob,          # absolute posterior for the winner
                loc_commit_margin,        # log-odds vs. runner-up
                loc_min_scans,            # evidence before we commit
                loc_rmse_sigma,           # for ICP likelihood
                loc_odom_sigma,           # m, for odometry consistency
                loc_odom_ang_sigma,
                loc_prune_thresh,
                loc_dup_dist,             # m, dedup threshold
                loc_dup_ang,
            )
        
        # debug
        self._verbose = verbose
    
    def process_scan(self, scan: PositionedCloud) -> tuple[np.ndarray | None, bool]:
        if self._use_prob_loc and not self._prob_loc_done:
            if len(self._saved_scans) > 0:
                res = self._hypothesis_localizer.find_on_map(scan)
                if res is not None:
                    if self._verbose:
                        print("prob_loc commit")
                        
                    self._commit_localization(*res)
                    self._hypothesis_localizer.reset()
                    self._prob_loc_done = True
                
                    return transform_4x4_to_2d_pose(self._current_global_pose), False
                
                elif self._loc_tries > self._loc_max_tries:
                    if self._verbose:
                        print("prob_loc timeout")
                        
                    self._prob_loc_done = True
                
                else:
                    if self._verbose:
                        print("prob_loc tried")
                        
                    self._loc_tries += 1
                    return transform_4x4_to_2d_pose(self._current_global_pose), False

            else:
                if self._verbose:
                    print("prob_loc empty hypothesis")
                    
                self._prob_loc_done = True
        
        # process scan        
        pos, closed = self._process_scan(scan)
        
        # redraw map from keyframes only if loop closed
        if closed:
            self._pending_scans.clear()
            self.map_builder.rebuild(self._saved_scans.values())

        # add a scan to drawing list
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
            
            self._current_global_pose = pose2d_to_transform(scan.estimated_pos)

            self._prev_scan = scan
            self._last_vertex_scan = scan
            self._last_subkeyframe = scan

            pose_id = self._graph.create_vertex(scan.pos, None)
            self._saved_scans[pose_id] = scan
            
            self._scandb.add_vertex_scan(
                pose_id,
                scan,
            )

            return None, False

        # correct using ICP if moved too much from last sub-keyframe or scans correction expired
        from_skf = np.linalg.norm(relative_pose(self._last_subkeyframe.pos, scan.pos)[:2])
        if from_skf > self._min_distance_to_subvertex or self._scans_no_correction >= self._max_scans_no_correction:
            self._scans_no_correction = 0

            # estimate transform from previous sub-keyframe to current
            T_sub_current, (fitness, rmse) = self._matcher.match(
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
                T_prevVertex_current, (fitness, rmse) = self._matcher.match(
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
                
                # ...but if match is really good - we can use estimated ICP pose delta to be more accurate
                if fitness >= self._vertex_acc_fitness and rmse < self._vertex_rmse_tol:
                    transf_2d = transform_4x4_to_2d_pose(T_prevVertex_current)
                    
                    transf_err = np.linalg.norm((transf_2d[:2] - rel_pose[:2]))
                    transf_err_rel = transf_err / max(np.linalg.norm(rel_pose[:2]), 0.1)
                    rot_err = abs(normalize_angle(transf_2d[2] - rel_pose[2]))
                    if transf_err_rel < self._vert_match_dist_tol and rot_err < self._vert_match_ang_tol:
                        tr = transf_2d

                # add vertex and register scan
                # pose_id = self._add_odom_pose(
                pose_id = self._graph.create_vertex(
                    pose_2d,
                    tr
                )
                self._scandb.add_vertex_scan(
                    pose_id,
                    scan,
                )
                
                self._saved_scans[pose_id] = scan
                self._last_vertex_scan = scan

                # ...and try to close loop
                closed = self.try_to_close_loop()
            
            return pose_2d, closed
        
        else:
            # just integrate position by odometry
            return self._proc_fallback_odom(scan)
            
    def _proc_fallback_odom(self, scan: PositionedCloud):
        # if there is no previous scan - nothing to calculate
        if self._prev_scan is None:
            return None, False

        # used to prevent odometry drift when staying still
        d = relative_pose(self._prev_scan.pos, scan.pos)
        trans = np.linalg.norm(d[:2])
        rot = abs(d[2])
        if trans < self._noise_trans_threshold and rot < self._noise_rot_threshold:
            scan.estimated_pos = self._prev_scan.estimated_pos
            self._prev_scan = scan
            return scan.estimated_pos, False

        self._scans_no_correction += 1

        # update global pose using estimated delta
        T_prev_current = relative_pose_4x4(
            self._prev_scan.pos,
            scan.pos
        )
        self._current_global_pose = self._current_global_pose @ T_prev_current
        scan.estimated_pos = transform_4x4_to_2d_pose(self._current_global_pose)
        self._prev_scan = scan

        return scan.estimated_pos, False

    def try_to_close_loop(self) -> bool:
        if self._prev_scan is None or self._prev_scan.estimated_pos is None:
            return False

        if self._verbose:
            print("trying to close loop")

        # ids are always in distance-ascending order
        coarse_ids, _ = self._scandb.search_candidates(self._prev_scan, self._coarse_search_max_candidates, self._min_chain_size)
        blacklist = get_topological_neighbors(
            self._graph.last_vertex_id, 
            self._graph.edges, 
            self._min_chain_size
        )

        vertices = [self._graph.vertices[id] for id in coarse_ids if id not in blacklist]

        # get the best match
        scans = [self._saved_scans[v.id] for v in vertices]
        idx, pose, (fitness, rmse) = self._matcher.find_best_match(
            scans, 
            self._prev_scan, 
            self._loop_acc_fitness, 
            self._vertex_rmse_tol
        )

        # if match found - calculate displacement error and 
        # close the loop if it is acceptable
        if pose is not None and idx >= 0:
            predicted = relative_from_vertices(self._graph.vertices[self._graph.last_vertex_id], vertices[idx])

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
                weight = fitness * np.exp(-rmse / 0.05)

                if self._verbose:
                    print("loop closure stats")
                    print("pose_loop pose_prev", vertices[idx].pose.position, self._graph.vertices[self._graph.last_vertex_id].pose.position)
                    print("fitness:", fitness, "rmse:", rmse)
                    print("weight:", weight)

                self._graph.create_edge(
                    self._graph.last_vertex_id,
                    vertices[idx].id, 
                    pose[:2], 
                    pose[2], 
                    self._loop_closure_edge_uncertainty / np.clip(weight, 0.1, 2.0)
                )

                if self._verbose:
                    print("added edge [", vertices[idx].id, self._graph.last_vertex_id, "]")

                self._graph.optimize()
                
                # update vertex values
                for vertex in self._graph.vertices.values():
                    self._saved_scans[vertex.id].estimated_pos = vertex.pose
                
                # update current global pos
                self._current_global_pose = pose2d_to_transform(self._graph.get_pose())

                return True

        if self._verbose:
            print("closure failed")

        return False

    def create_occupancy_grid(self, occupied_thresh: float = 0.4) -> OccupancyGrid:
        if len(self._pending_scans) > 0:
            self.map_builder.add_scans(self._pending_scans)
            self._pending_scans.clear()

        return self.map_builder.to_occupancy_grid(occupied_thresh)

    def new_session(self, initial_position: np.ndarray | None):
        """
        start a new SLAM localization session
        resets all session-bound variables, including position
        
        starts probabilistic pose search if the *initial_position* is None 
        
        @param initial_position: x, y, theta
        """
        
        self._last_vertex_scan = None
        self._prev_scan = None
        self._last_subkeyframe = None
        self._scans_no_correction = 0
        
        if initial_position is None:
            self._prob_loc_done = False
        
        else:
            self._current_global_pose = pose2d_to_transform(initial_position)

    def _commit_localization(
        self, h: LocalizationHypothesis, scan: PositionedCloud
    ) -> None:
        """Lock the winning hypothesis in and reseed the SLAM state from it."""
        scan.estimated_pos = h.pose
        self._current_global_pose = pose2d_to_transform(h.pose)

        self._prev_scan             = scan
        self._last_vertex_scan      = scan
        self._last_subkeyframe      = scan
        self._scans_no_correction   = 0

        # Anchor the graph at the localized pose and register the scan.
        pose_id = self._graph.create_vertex(h.pose, None)
        self._saved_scans[pose_id] = scan
        self._scandb.add_vertex_scan(pose_id, scan)

        # Optional: connect it to the reference vertex with a strong prior so
        # the graph is not disconnected.  Direction: from new -> ref.
        if h.vertex_id != pose_id:
            ref_pose = self._graph.vertices[h.vertex_id].pose
            rel = relative_pose(h.pose, ref_pose)   # T_new_ref in 2D
            self._graph.create_edge(
                pose_id,
                h.vertex_id,
                rel[:2],
                rel[2],
                self._loop_closure_edge_uncertainty / 2.0,
            )
            
            self._graph.optimize()
            for vertex in self._graph.vertices.values():
                self._saved_scans[vertex.id].estimated_pos = vertex.pose

        # Rebuild the map from keyframes now that the pose is trustworthy.
        self.map_builder.rebuild(self._saved_scans.values())

    def save_to_file(self, fp: str) -> None:
        # сбрасываем pending в карту, иначе потеряем последние ~8 сканов
        if self._pending_scans:
            self.map_builder.add_scans(self._pending_scans)
            self._pending_scans.clear()

        # PoseSE2 не всегда итерируется — достаём явно
        def pose_to_list(p) -> list[float]:
            return [float(p.position[0]), float(p.position[1]), float(p.orientation)]

        graph = self._graph

        vertices = [
            [int(vid), *pose_to_list(v.pose)]
            for vid, v in graph.vertices.items()
        ]

        edges = [
            [
                list(map(int, e.vertex_ids)),
                pose_to_list(e.estimate),
                np.asarray(e.information, dtype=np.float64).flatten().tolist(),
            ]
            for e in graph.edges
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
            "dxy_var": graph._dxy_var,
            "dtheta_var": graph._dtheta_var,
            "odometry_edge_uncertainty": self._odometry_edge_uncertainty,
            "loop_closure_edge_uncertainty": self._loop_closure_edge_uncertainty,

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
            "next_vertex_id": int(graph.next_vertex_id),
            "scans_no_correction": int(self._scans_no_correction),

            # --- graph data ---
            "graph_vertices": vertices,
            "graph_edges": edges,
        }

        params_encoded  = json.dumps(parameters, separators=(',', ':')).encode("utf-8")
        scandb_encoded  = self._scandb.to_bytes()
        scans_encoded   = pickle.dumps(
            {int(k): v.to_bytes() for k, v in self._saved_scans.items()},
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        with open(fp, "wb") as f:
            for blob in (params_encoded, scandb_encoded, scans_encoded):
                f.write(struct.pack(">I", len(blob)))
                f.write(blob)

    @staticmethod
    def load_from_file(fp: str) -> "VGSLAM":
        with open(fp, "rb") as f:
            def read_blob() -> bytes:
                n = struct.unpack(">I", f.read(4))[0]
                return f.read(n)

            params_encoded  = read_blob()
            scandb_encoded  = read_blob()
            scans_encoded   = read_blob()

        parameters = json.loads(params_encoded.decode("utf-8"))
        scans_raw  = pickle.loads(scans_encoded)      # dict[int, bytes]

        # 1. создаём SLAM с сохранёнными параметрами
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
            odometry_edge_uncertainty=parameters["odometry_edge_uncertainty"],
            loop_closure_edge_uncertainty=parameters["loop_closure_edge_uncertainty"],

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

        # 2. восстанавливаем вершины напрямую в _graph._vertices
        for v_data in parameters["graph_vertices"]:
            vid    = int(v_data[0])
            x, y, theta = float(v_data[1]), float(v_data[2]), float(v_data[3])
            slam._graph._vertices[vid] = Vertex(
                vertex_id=vid,
                pose=PoseSE2(position=np.array([x, y], dtype=np.float64),
                            orientation=theta),
                fixed=(vid == 0),
            )

        # 3. восстанавливаем рёбра (с information-матрицей)
        for e_data in parameters["graph_edges"]:
            ids        = list(map(int, e_data[0]))
            x, y, theta = float(e_data[1][0]), float(e_data[1][1]), float(e_data[1][2])
            info       = np.asarray(e_data[2], dtype=np.float64).reshape(3, 3)

            slam._graph._edges.append(EdgeOdometry(
                vertex_ids=ids,
                information=info,
                estimate=PoseSE2(position=np.array([x, y], dtype=np.float64),
                                orientation=theta),
            ))

        # 4. счётчики
        slam._graph.next_vertex_id = int(parameters["next_vertex_id"])
        slam._scans_no_correction  = int(parameters["scans_no_correction"])

        # 5. ScanDB + сохранённые сканы
        slam._scandb = ScanDB.from_bytes(scandb_encoded)
        slam._saved_scans = {
            int(k): PositionedCloud.from_bytes(v)
            for k, v in scans_raw.items()
        }

        # 6. нормализуем estimated_pos — после сохранения мог остаться PoseSE2
        for scan in slam._saved_scans.values():
            ep = scan.estimated_pos
            if ep is not None and not isinstance(ep, np.ndarray):
                scan.estimated_pos = np.array(
                    [ep.position[0], ep.position[1], ep.orientation],
                    dtype=np.float64,
                )

        # 7. восстанавливаем «текущие» указатели из последнего keyframe'а
        if slam._graph.next_vertex_id > 0:
            last_id   = slam._graph.next_vertex_id - 1
            last_scan = slam._saved_scans[last_id]

            slam._last_vertex_scan = last_scan
            slam._prev_scan        = last_scan
            slam._last_subkeyframe = last_scan

            slam._current_global_pose = pose2d_to_transform(last_scan.estimated_pos) # type: ignore

        # 8. пересобираем карту по keyframe'ам — грид детерминирован
        slam.map_builder.rebuild(slam._saved_scans.values())
        
        # 9
        if slam._use_prob_loc:
            slam._hypothesis_localizer = LocalizationHypothesisSelection(
                slam._saved_scans,
                slam._matcher,
                slam._graph,
                slam._scandb,
            )
            slam._prob_loc_done = False
            slam._loc_tries = 0

        return slam
