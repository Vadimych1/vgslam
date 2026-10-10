from .helpers import transform_4x4_to_2d_pose, relative_pose_4x4
import numpy as np
import small_gicp    
import struct

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
    def __init__(self, cloud: np.ndarray, pos: np.ndarray, estimated_pos: np.ndarray | None = None) -> None:
        self.cloud = cloud

        self.pos = np.array(pos)
        self.estimated_pos = estimated_pos
        
        self._gicp = None
        self._kdtree = None
        self._sc_kdtree = None
        self._transform_mat = None
        
        self._descriptor: np.ndarray | None = None
        
    def relative(self, other):   # T_self_other в оценочных позах
        assert self.estimated_pos is not None and other.estimated_pos is not None
        return relative_pose_4x4(self.estimated_pos, other.estimated_pos)

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
    
        if self._gicp is None:
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

    def to_bytes(self) -> bytes:
        buf = bytearray()

        # --- cloud: (N, 3) float32 ---
        cloud = np.ascontiguousarray(self.cloud, dtype=np.float32)
        assert cloud.ndim == 2 and cloud.shape[1] == 3, \
            f"expected cloud (N, 3), got {cloud.shape}"
        buf += struct.pack(">II", cloud.shape[0], cloud.shape[1])
        buf += cloud.tobytes()

        # --- pos: (3,) float64 ---
        pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        assert pos.shape == (3,), f"expected pos (3,), got {pos.shape}"
        buf += pos.tobytes()

        # --- estimated_pos: (3,) float64 | None ---
        if self.estimated_pos is None:
            buf += struct.pack(">B", 0)
        else:
            ep = np.ascontiguousarray(self.estimated_pos, dtype=np.float64)
            assert ep.shape == (3,), f"expected estimated_pos (3,), got {ep.shape}"
            buf += struct.pack(">B", 1)
            buf += ep.tobytes()

        # --- descriptor: (192,) float32 | None ---
        if self._descriptor is None:
            buf += struct.pack(">B", 0)
        else:
            desc = np.ascontiguousarray(self._descriptor, dtype=np.float32).ravel()
            buf += struct.pack(">B", 1)
            buf += struct.pack(">I", desc.size)
            buf += desc.tobytes()

        return bytes(buf)

    @staticmethod
    def from_bytes(data: bytes) -> "PositionedCloud":
        off = 0

        # --- cloud ---
        n, m = struct.unpack_from(">II", data, off); off += 8
        cloud = np.frombuffer(data[off:off + n * m * 4], dtype=np.float32).copy()
        cloud = cloud.reshape(n, m)
        off += n * m * 4

        # --- pos ---
        pos = np.frombuffer(data[off:off + 24], dtype=np.float64).copy()
        off += 24

        # --- estimated_pos ---
        flag = struct.unpack_from(">B", data, off)[0]; off += 1
        if flag:
            estimated_pos = np.frombuffer(data[off:off + 24], dtype=np.float64).copy()
            off += 24
        else:
            estimated_pos = None

        # --- descriptor ---
        flag = struct.unpack_from(">B", data, off)[0]; off += 1
        descriptor = None
        if flag:
            dsize = struct.unpack_from(">I", data, off)[0]; off += 4
            descriptor = np.frombuffer(data[off:off + dsize * 4], dtype=np.float32).copy()
            off += dsize * 4

        obj = PositionedCloud(cloud=cloud, pos=pos, estimated_pos=estimated_pos)
        obj._descriptor = descriptor
        return obj
