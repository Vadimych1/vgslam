from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se2 import PoseSE2
from graphslam.graph import Graph

import numpy as np

from .helpers import pose2d_to_transform

class PoseGraph:
    def __init__(
        self,
        dxy_var: float = 0.2, 
        dtheta_var: float = np.deg2rad(3),
        loop_closure_edge_uncertainty: float = 0.5,
        odometry_edge_uncertainty: float = 1.0,
    ):
        self.next_vertex_id = 0
        self._vertices: dict[int, Vertex] = {}
        self._edges: list[EdgeOdometry] = []

        self._dxy_var = dxy_var
        self._dtheta_var = dtheta_var
        self._graph = Graph([], [])
        self._loop_closure_edge_uncertainty = loop_closure_edge_uncertainty
        self._odometry_edge_uncertainty = odometry_edge_uncertainty
    
    @property
    def vertices(self) -> dict[int, Vertex]:
        return self._vertices
    
    @property
    def edges(self) -> list[EdgeOdometry]:
        return self._edges
    
    @property
    def last_vertex_id(self) -> int:
        return self.next_vertex_id - 1
    
    def create_vertex(self, est_pos: np.ndarray, dpos: np.ndarray | None):
        # add a new vertex
        self._vertices[self.next_vertex_id] = Vertex(
            vertex_id=self.next_vertex_id,
            pose=PoseSE2(
                position=est_pos[:2],
                orientation=est_pos[2]
            ),
            fixed=self.next_vertex_id == 0
        )
        
        # and make a new edge if dpos (relative pose) is provided
        if self.next_vertex_id > 0 and dpos is not None:
            self.create_edge(
                self.next_vertex_id - 1,
                self.next_vertex_id,
                dpos[:2],
                dpos[2],
                self._odometry_edge_uncertainty
            )
        
        self.next_vertex_id += 1
        return self.next_vertex_id - 1
    
    def create_edge(self, id_a: int, id_b: int, odom_dpos: np.ndarray, odom_dtheta: float, var_mul: float):
        # add an edge with variances
        self._edges.append(EdgeOdometry(vertex_ids=[id_a, id_b], information=np.array([
            [1 / (self._dxy_var * var_mul), 0, 0],
            [0, 1 / (self._dxy_var * var_mul), 0],
            [0, 0, 1 / (self._dtheta_var * var_mul)],
        ]), estimate=PoseSE2(
            position=odom_dpos,
            orientation=odom_dtheta
        )))
        
    def optimize(self):
        self._graph = Graph(
            self._edges,
            list(self._vertices.values())
        )
        self._graph.optimize(verbose=False)

    def get_pose(self, idx: int | None = None):
        if idx == None:
            return self._vertices[self.last_vertex_id].pose

        else:
            return self._vertices[idx].pose
