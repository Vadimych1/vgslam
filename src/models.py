from .helpers import transform_4x4_to_2d_pose
import numpy as np
import small_gicp

class PositionedCloud:
    def __init__(self, cloud: np.ndarray, pos: list[float] | np.ndarray, estimated_pos: list[float] | np.ndarray | None = None) -> None:
        self.cloud = cloud

        self.pos = np.array(pos)
        self.estimated_pos = estimated_pos
        
        self._gicp = None
        self._kdtree = None
        self._transform_mat = None
        
    def relative(self, other):
        x1, y1, t1 = other.pos if other.estimated_pos is None else other.estimated_pos
        x2, y2, t2 = self.pos if self.estimated_pos is None else self.estimated_pos

        dx_world = x2 - x1
        dy_world = y2 - y1

        c = np.cos(t1)
        s = np.sin(t1)

        dx =  c * dx_world + s * dy_world
        dy = -s * dx_world + c * dy_world

        dtheta = np.arctan2(
            np.sin(t2 - t1),
            np.cos(t2 - t1)
        )

        return np.array([
            [np.cos(dtheta), -np.sin(dtheta), 0, dx],
            [np.sin(dtheta),  np.cos(dtheta), 0, dy],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def gicp(self) -> small_gicp.PointCloud:
        """
        Returns small_gicp point cloud format
        Automatically estimates cloud covariances when called for the first time
        """
        
        if self._gicp is None:
            self._gicp = small_gicp.PointCloud(self.cloud)
            small_gicp.estimate_covariances(self._gicp)
        
        return self._gicp
    
    def kdtree(self) -> small_gicp.KdTree:
        if self._kdtree is None:
            self._kdtree = small_gicp.KdTree(self.gicp())
        
        return self._kdtree
    
    def set_pose(self, transform_4x4: np.ndarray):
        self._transform_mat = transform_4x4
        self.estimated_pos = transform_4x4_to_2d_pose(transform_4x4)