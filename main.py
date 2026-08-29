from src.helpers import *
from src.models import *
from src.sim import RobotSimulator

import numpy as np
import cv2 as cv
# from point_cloud_registration import ICP, NDT
import small_gicp as sg

from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se2 import PoseSE2
from graphslam.graph import Graph

any_iter = list[float] | np.ndarray

class SLAM:
    def __init__(self, dxy_covar: float, dtheta_covar: float, min_distance_to_vertex: float, min_chain_size: float, loop_closure_max_dist: float, verbose: bool = True) -> None:
        self.next_vertex_id = 0
    
        self.vertices: list[Vertex] = []
        self.edges: list[EdgeOdometry] = []
        
        self._dxy_covar = dxy_covar
        self._dtheta_covar = dtheta_covar
        
        self._graph = Graph([], [])
        self._vertex_id_to_neighbors = {}
        
        self._min_distance_to_vertex = min_distance_to_vertex
        self._min_chain_size = min_chain_size
        self._vertices_no_closure = 0
        self._loop_closure_max_dist = loop_closure_max_dist

        self._current_global_pose = np.eye(4)
        self._last_vertex_scan = None
        self._prev_scan = None
        self._saved_scans: dict[int, PositionedCloud] = {}
        
        self._verbose = verbose
        
    def process_scan(self, scan: PositionedCloud) -> tuple[any_iter | None, bool]:
        if self._prev_scan is None or self._last_vertex_scan is None:
            scan.estimated_pos = scan.pos
            
            self._prev_scan = scan
            self._last_vertex_scan = scan
            
            pose_id = self._add_odom_pose(scan.pos, [0, 0, 0])
            self._saved_scans[pose_id] = scan
        
            return None, False
        
        T_prev_current = self._process_scans(scan, self._prev_scan)
        
        self._current_global_pose  = self._current_global_pose @ T_prev_current
        
        pose_2d = transform_4x4_to_2d_pose(self._current_global_pose)
        
        scan.estimated_pos = pose_2d
        self._prev_scan = scan
        
        closed = False
        if np.linalg.norm((pose_2d - self._last_vertex_scan.estimated_pos)[:2]) >= self._min_distance_to_vertex:
            self._vertices_no_closure += 1
            
            pose_id = self._add_odom_pose(
                pose_2d,
                relative_pose(self._last_vertex_scan.estimated_pos, scan.estimated_pos)
            )
            
            self._saved_scans[pose_id] = scan
            self._last_vertex_scan = scan
            
            if self._vertices_no_closure >= self._min_chain_size:            
                closed = self.try_to_close_loop()
    
                if closed:
                    self._vertices_no_closure = 0
        
        return pose_2d, closed
        
    def _process_scans(self, source: PositionedCloud, target: PositionedCloud):
        pcd_src = source.gicp()
        pcd_tgt = target.gicp()
        tree = target.kdtree()
        
        result = small_gicp.align(
            pcd_tgt, pcd_src, tree,
            np.eye(4),
            registration_type="GICP"
        )
        
        pose = result.T_target_source
        return pose
    
    def _find_best_match(self, sources: list[PositionedCloud], target: PositionedCloud, min_confidence = 0.9, rmse_max = 0.1) -> tuple[int, any_iter | None, tuple]:
        for i, source in enumerate(sources):
            T_new = self._process_scans(source, target)
            fitness, inlier_rmse = evaluate_registration(
                source.cloud,
                target.cloud,
                T_new,
                max_correspondence_distance=0.2
            )
            
            if fitness > min_confidence and inlier_rmse < rmse_max:
                return i, transform_4x4_to_2d_pose(T_new), (fitness, inlier_rmse)
        
        return -1, None, (0, 0)
    
    def _add_odom_pose(self, est_pos: any_iter, odom_dpos: any_iter | None):
        """
        Adds odometry vertex to pose graph
        Creates an edge to previously added vertex if `odom_dpos` is not None
        
        @param est_pos: estimated position
        @param odom_dpos: odometry delta from previous pose
        @return: id of added vertex
        """        
        self.vertices.append(Vertex(
            vertex_id=self.next_vertex_id,
            pose=PoseSE2(
                position=est_pos[:2],
                orientation=est_pos[2]
            ),
            fixed=self.next_vertex_id == 0
        ))
        
        self._vertex_id_to_neighbors[self.next_vertex_id] = []
        
        if self.next_vertex_id > 0 and odom_dpos is not None:
            self._add_edge(
                self.next_vertex_id - 1,
                self.next_vertex_id,
                odom_dpos[:2],
                odom_dpos[2],
                1.0
            )
            
            self._vertex_id_to_neighbors[self.next_vertex_id].append(self.next_vertex_id - 1)
            self._vertex_id_to_neighbors[self.next_vertex_id - 1].append(self.next_vertex_id)
        
        self.next_vertex_id += 1
        return self.next_vertex_id - 1
    
    def _add_edge(self, id_a: int, id_b: int, odom_dpos: list[float], odom_dtheta: float, covar_mul: float):
        self.edges.append(EdgeOdometry(vertex_ids=[id_a, id_b], information=np.array([
            [1 / (self._dxy_covar * covar_mul), 0, 0],
            [0, 1 / (self._dxy_covar * covar_mul), 0],
            [0, 0, 1 / (self._dtheta_covar * covar_mul)],
        ]), estimate=PoseSE2(
            position=odom_dpos,
            orientation=odom_dtheta
        )))

    def try_to_close_loop(self) -> bool:
        if self._prev_scan is None or self._prev_scan.estimated_pos is None:
            return False
        
        if self._verbose:
            print("trying to close loop")
        
        blacklist = get_topological_neighbors(self.next_vertex_id - 1, self.edges, 6) # TODO: move this to config
        vertices_with_dist = [(v, np.linalg.norm(np.array(v.pose.position) - np.array(self._prev_scan.estimated_pos[:2]))) for v in self.vertices if v.id not in blacklist]
        nearest_vertices = [v for v, dist in sorted(vertices_with_dist, key=lambda x: x[1]) if dist <= self._loop_closure_max_dist]
        
        scans = [self._saved_scans[v.id] for v in nearest_vertices]
        idx, pose, (fitness, rmse) = self._find_best_match(scans, self._prev_scan)
        
        if pose is not None and idx >= 0:
            p_x, p_y, p_t = pose

            if self._verbose:
                print("loop closure stats")
                print(nearest_vertices[idx].pose.position, self.vertices[self.next_vertex_id - 1].pose.position)
                print(relative_from_vertices(nearest_vertices[idx], self.vertices[self.next_vertex_id - 1]))
                print(p_x, p_y, p_t)
                print("fitness:", fitness, "rmse:", rmse)
            
            self._add_edge(self.next_vertex_id - 1, nearest_vertices[idx].id, [p_x, p_y], p_t, 2.0)
            
            print("adding edge", nearest_vertices[idx].id, self.next_vertex_id - 1)
            
            self.optimize()
            
            if self._verbose:
                print("closed and optimized")

            return True
        
        if self._verbose:
            print("closure failed")
            
        return False
    
    def _world_to_grid(self, points: np.ndarray, min_x: float, min_y: float, resolution: float, cvt=True) -> np.ndarray:
        g = np.floor((points - np.array([min_x, min_y])) / resolution)
        
        if cvt:
            return g.astype(np.int64)
        
        return g
    
    def create_occupancy_grid(self, resolution: float = 0.05):
        perscan_points = []
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
        
        all_points = np.vstack(perscan_points)
        
        min_x = all_points[:, 0].min()
        max_x = all_points[:, 0].max()
        min_y = all_points[:, 1].min()
        max_y = all_points[:, 1].max()
        
        width = int(np.ceil((max_x - min_x) / resolution)) + 1
        height = int(np.ceil((max_y - min_y) / resolution)) + 1
        
        grid = np.full(
            (height, width),
            127,
            dtype=np.uint8,
        )
        
        for points, scan in zip(perscan_points, self._saved_scans.values()):
            g = self._world_to_grid(
                points,
                min_x,
                min_y,
                resolution
            )
            
            rx, ry, _ = scan.estimated_pos if scan.estimated_pos is not None else scan.pos
            rx, ry = self._world_to_grid(np.array([rx, ry]), min_x, min_y, resolution)
            
            for px, py in g:
                if 0 <= px < width and 0 <= py < height:
                    for fx, fy in self._bresenham(px, py, rx, ry):
                        if 0 <= fx < width and 0 <= fy < height and grid[fy, fx] != 0:
                            grid[fy, fx] = 255
                            
                    grid[py, px] = 0
            
        return grid
    
    def _bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:
            yield x0, y0

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x0 += sx

            if e2 < dx:
                err += dx
                y0 += sy
            
    def optimize(self):
        self._graph._vertices = self.vertices
        self._graph._edges = self.edges
        self._graph._initialize()

        res = self._graph.optimize(verbose=False)
        
        for vertex in self.vertices:
            self._saved_scans[vertex.id].estimated_pos = vertex.pose
        
        print("optimized graph in", res.duration_s)

    def _recalculate_graph(self):
        ...

sim = RobotSimulator(
    dt=0.1,
    lidar_angle_step=np.deg2rad(0.5)
)
slam = SLAM(4, 3, 0.2, 6, 0.7, False)

# scan1 = sim.step(speed=0.5)
# scan2 = sim.step(speed=0.5)

# d = slam._process_scans(
#     PositionedCloud(
#         scan_to_cloud(scan2.ranges, scan2.angles, 12),
#         scan2.odom_pos,
#     ),
#     PositionedCloud(
#         scan_to_cloud(scan1.ranges, scan1.angles, 12),
#         scan1.odom_pos,
#     ),
# )

# print(d)
# print(scan2.true_pos - scan1.true_pos)
# print(scan2.odom_pos - scan1.odom_pos)

# quit()

for _ in range(300):
    scan = sim.step(speed=0.5)
    pos, closed = slam.process_scan(
        PositionedCloud(
            scan_to_cloud(scan.ranges, scan.angles, 12),
            scan.odom_pos,
        )
    )
    
    # if pos is not None:
        # print(np.array(pos) - np.array(scan.true_pos))
        # print(scan.true_pos, "|", pos)

save_slam_graph(slam.vertices, slam.edges)
cv.imwrite("map.png", slam.create_occupancy_grid(0.05))
