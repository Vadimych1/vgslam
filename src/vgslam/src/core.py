import numpy as np
import small_gicp

from .helpers import *
from .models import *
from .scandb import ScanDB

from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se2 import PoseSE2
from graphslam.graph import Graph

from typing import Literal
from numba.typed.typedlist import List

class VGSLAM:
    def __init__(
        self, 
        # graph optimization
        dxy_var: float = 0.2, 
        dtheta_var: float = np.deg2rad(3), 
        
        # loop closure
        min_chain_size: int = 5, 
        loop_closure_max_dist: float = 0.7,
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
        
        # icp
        matcher_max_correspondence_distance: float = 0.5,
        matcher_mode: Literal['ICP', 'GICP', 'PLANE_ICP'] = "GICP",
        
        # debugging log
        verbose: bool = False
    ) -> None:
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
        self._matcher_mode = matcher_mode
        
        self._verbose = verbose
    
    def process_scan(self, scan: PositionedCloud) -> tuple[np.ndarray | None, bool]:
        """
        Process new lidar scan with odometry
        """
        
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
                scan.estimated_pos = transform_4x4_to_2d_pose(self._current_global_pose)
                self._last_subkeyframe = scan
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
                    rot_err = abs(normalize_angle(transf_2d[2] - rel_pose[2]))
                    if transf_err < self._vert_match_dist_tol and rot_err < self._vert_match_ang_tol:
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
        
        self._scans_no_correction += 1
        
        # get delta transform and apply it on the global pose
        prev_pos = self._prev_scan.pos
        dx_rel, dy_rel, dtheta_rel = relative_pose(scan.pos, prev_pos)
        _, _, yaw_g = -transform_4x4_to_2d_pose(self._current_global_pose)
        
        dx_g = np.cos(yaw_g) * dx_rel - np.sin(yaw_g) * dy_rel
        dy_g = np.sin(yaw_g) * dx_rel + np.cos(yaw_g) * dy_rel
        dtheta_g = dtheta_rel
        
        T_prev_current = pose2d_to_transform(np.array([dx_g, dy_g, dtheta_g]))
        self._current_global_pose = self._current_global_pose @ T_prev_current

        scan.estimated_pos = transform_4x4_to_2d_pose(self._current_global_pose)
        self._prev_scan = scan
        
        return transform_4x4_to_2d_pose(self._current_global_pose), False
        
    def _process_scans(self, source: PositionedCloud, target: PositionedCloud, rel_override: np.ndarray | None = None):
        # grab cached or newly created scan data
        pcd_src = source.gicp()
        pcd_tgt = target.gicp()
        tree = target.kdtree()
        
        result = small_gicp.align(
            pcd_tgt, pcd_src, tree,
            relative_pose_4x4(source.pos, target.pos) if rel_override is None else rel_override,
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
        # iterate over scans and return the first match that has good scores
        # source array should be sorted by match probability for best results
        for i, source in enumerate(sources):
            T_new, (fitness, inlier_rmse) = self._process_scans(source, target)
            
            if fitness > min_confidence and inlier_rmse < rmse_max:
                return i, transform_4x4_to_2d_pose(T_new), (fitness, inlier_rmse)
        
        return -1, None, (0, 0)
    
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
        
        # filter vertices by distance to current
        cur_pose = np.array(self._prev_scan.estimated_pos[:2])
        vertices_with_dist_sq = [(v, np.sum((np.array(v.pose.position) - cur_pose) ** 2)) for v in vertices]
        nearest_vertices = [v for v, dist in vertices_with_dist_sq if dist <= self._loop_closure_max_dist ** 2]
        
        # get the best match
        scans = [self._saved_scans[v.id] for v in nearest_vertices]
        idx, pose, (fitness, rmse) = self._find_best_match(scans, self._prev_scan, self._loop_acc_fitness, self._vertex_rmse_tol)
        
        # if match found - calculate displacement error and 
        # close the loop if it is acceptable
        if pose is not None and idx >= 0:
            predicted = relative_from_vertices(self.vertices[self.next_vertex_id - 1], nearest_vertices[idx])
            
            err = np.linalg.norm((pose - predicted)[:2])
            
            angle_err = pose[2] - predicted[2]
            angle_err = abs(normalize_angle(angle_err))
            
            if self._verbose:
                print("pose_err angle_err" , err, angle_err)
                print("predicted_pose estimated_pose", predicted, pose)
            
            if err <= 0.5 and angle_err <= np.deg2rad(30):
                if self._verbose:
                    print("loop closure stats")
                    print("pose_loop pose_prev", nearest_vertices[idx].pose.position, self.vertices[self.next_vertex_id - 1].pose.position)
                    print("fitness:", fitness, "rmse:", rmse)
                
                self._add_edge(self.next_vertex_id - 1, nearest_vertices[idx].id, pose[:2], pose[2], 0.5)
                
                if self._verbose:
                    print("added edge [", nearest_vertices[idx].id, self.next_vertex_id - 1, "]")
                
                self.optimize()

                return True
        
        if self._verbose:
            print("closure failed")
            
        return False
    
    def create_occupancy_grid(self, resolution: float = 0.05):
        perscan_points = List()
        robot_positions = []
        
        for scan in self._saved_scans.values():            
            x, y, theta = scan.estimated_pos if scan.estimated_pos is not None else scan.pos
    
            c = np.cos(theta)
            s = np.sin(theta)
            
            points = scan.cloud[:, :2]
            world = points @ np.array([
                [c, s],
                [-s, c],
            ]) + np.array([x, y])
            
            perscan_points.append(world)
            robot_positions.append((x, y))
            
        all_points = np.vstack(perscan_points)
        
        min_x = all_points[:, 0].min()
        max_x = all_points[:, 0].max()
        min_y = all_points[:, 1].min()
        max_y = all_points[:, 1].max()
        
        width = int(np.ceil((max_x - min_x) / resolution)) + 1
        height = int(np.ceil((max_y - min_y) / resolution)) + 1
        
        robot_positions = world_to_grid(np.array(robot_positions), min_x, min_y, resolution)
        
        grid = np.zeros(
            (height, width),
            dtype=np.float32,
        )
        
        update_scan(
            grid,
            perscan_points,
            robot_positions,
            min_x,
            min_y,
            resolution,
            width,
            height,
            0.85,
            0.4,
        )
        
        occ_grid = np.empty_like(grid, dtype=np.uint8)

        occ_grid[grid < 0] = 0
        occ_grid[grid == 0] = 127
        occ_grid[grid > 0] = 255
        
        return OccupancyGrid(
            occ_grid,
            -min_x, # world (0, 0) on map is located at (-min_x, -min_y)
            -min_y,
            width,
            height,
            resolution,
        )
            
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
