from .helpers import transform_4x4_to_2d_pose, relative_pose_4x4
import numpy as np
import small_gicp    
import io

class OccupancyGrid:
    def __init__(self, 
        grid: np.ndarray,
        zero_offset_x_px: int,
        zero_offset_y_px: int,
        width: int,
        height: int,
        resolution: float,
    ) -> None:
        self.grid = grid

        self.zero_offset_x_px = zero_offset_x_px
        self.zero_offset_y_px = zero_offset_y_px

        self.width = width
        self.height = height

        self.resolution = resolution

class PositionedCloud:
    def __init__(self, cloud: np.ndarray, pos: np.ndarray, estimated_pos: np.ndarray | None = None, resolution: float = 0.05) -> None:
        self.cloud = cloud

        self.pos = np.array(pos)
        self.estimated_pos = estimated_pos

        self._gicp = None
        self._kdtree = None
        self._sc_kdtree = None
        self._transform_mat = None

        self._descriptor: np.ndarray | None = None

        self._resolution = resolution

    def relative(self, other): # T_self_other in estimated poses
        assert self.estimated_pos is not None and other.estimated_pos is not None
        return relative_pose_4x4(self.estimated_pos, other.estimated_pos)

    @property
    def gicp(self) -> small_gicp.PointCloud:
        """
        Returns small_gicp point cloud format
        Automatically estimates cloud covariances when called for the first time
        """

        if self._gicp is None:
            self._gicp = small_gicp.voxelgrid_sampling(self.cloud, self._resolution)
            small_gicp.estimate_normals_covariances(self._gicp)

        return self._gicp
    
    @property
    def kdtree(self) -> small_gicp.KdTree:
        if self._kdtree is None:
            self._kdtree = small_gicp.KdTree(self.gicp)
        
        return self._kdtree
    
    def set_pose(self, transform_4x4: np.ndarray):
        self._transform_mat = transform_4x4
        self.estimated_pos = transform_4x4_to_2d_pose(transform_4x4)

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        data_dict = {
            "cloud": np.ascontiguousarray(self.cloud, dtype=np.float32),
            "pos": np.ascontiguousarray(self.pos, dtype=np.float64)
        }
        if self.estimated_pos is not None:
            data_dict["estimated_pos"] = np.ascontiguousarray(self.estimated_pos, dtype=np.float64)
        if self._descriptor is not None:
            data_dict["descriptor"] = np.ascontiguousarray(self._descriptor, dtype=np.float32)

        np.savez_compressed(buf, **data_dict)
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> "PositionedCloud":
        buf = io.BytesIO(data)
        with np.load(buf) as archive:
            cloud = archive["cloud"]
            pos = archive["pos"]
            estimated_pos = archive["estimated_pos"] if "estimated_pos" in archive else None
            descriptor = archive["descriptor"] if "descriptor" in archive else None

        obj = PositionedCloud(cloud=cloud, pos=pos, estimated_pos=estimated_pos)
        obj._descriptor = descriptor
        return obj
