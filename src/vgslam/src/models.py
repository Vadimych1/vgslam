from .helpers import transform_4x4_to_2d_pose, relative_pose_4x4
import numpy as np
import small_gicp
# import open3d as o3d


class PositionedCloud:
    def __init__(self, cloud: np.ndarray, pos: np.ndarray, estimated_pos: np.ndarray | None = None) -> None:
        self.cloud = cloud

        self.pos = np.array(pos)
        self.estimated_pos = estimated_pos
        
        self._gicp = None
        self._kdtree = None
        self._sc_kdtree = None
        self._transform_mat = None
        
        self._descriptor: np.ndarray | None = None
        
    def relative(self, other):
        p1 = other.pos if other.estimated_pos is None else other.estimated_pos
        p2 = self.pos if self.estimated_pos is None else self.estimated_pos

        return relative_pose_4x4(p1, p2)

    def gicp(self, voxel_downsample: bool = False, voxel_size: float = 0.05) -> small_gicp.PointCloud:
        """
        Returns small_gicp point cloud format
        Automatically estimates cloud covariances when called for the first time
        """
    
        # TODO: fix downsampling issues    
        # if self._gicp is None and voxel_downsample:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(self.cloud)
        #     downsampled = pcd.voxel_down_sample(voxel_size)
            
        #     self._gicp = small_gicp.PointCloud(np.asarray(downsampled.points))
        #     small_gicp.estimate_normals_covariances(self._gicp)
            
        # elif self._gicp is None:
    
        self._gicp = small_gicp.PointCloud(self.cloud)
        small_gicp.estimate_normals_covariances(self._gicp)
    
        return self._gicp
    
    def kdtree(self) -> small_gicp.KdTree:
        if self._kdtree is None:
            self._kdtree = small_gicp.KdTree(self.gicp())
        
        return self._kdtree
    
    def set_pose(self, transform_4x4: np.ndarray):
        self._transform_mat = transform_4x4
        self.estimated_pos = transform_4x4_to_2d_pose(transform_4x4)
    
    