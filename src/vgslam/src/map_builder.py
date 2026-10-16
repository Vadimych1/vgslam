import numpy as np
import numba
from .models import PositionedCloud, OccupancyGrid
from typing import Iterable

@numba.njit(cache=True)
def _update_scan_batch(
    grid: np.ndarray,            # (H, W)
    flat_points: np.ndarray,     # (M, 2)
    offsets: np.ndarray,         # (N+1,)
    robot_positions: np.ndarray, # (N, 2)
    min_x: float,
    min_y: float,
    inv_resolution: float,
    width: int,
    height: int,
    occ_dec: float,
    free_inc: float,
    l_min: float,
    l_max: float
):
    for scan_idx in range(robot_positions.shape[0]):
        rx = robot_positions[scan_idx, 0]
        ry = robot_positions[scan_idx, 1]
        
        if rx < 0:
            rx = 0
        elif rx >= width:
            rx = width - 1
        
        if ry < 0:
            ry = 0
        elif ry >= height:
            ry = height - 1
        
        start = offsets[scan_idx]
        end = offsets[scan_idx + 1]
        
        for i in range(start, end):
            px0 = int(np.floor((flat_points[i, 0] - min_x) * inv_resolution))
            py0 = int(np.floor((flat_points[i, 1] - min_y) * inv_resolution))
            
            if 0 <= px0 < width and 0 <= py0 < height:
                v = grid[py0, px0] - occ_dec
                if v < l_min:
                    v = l_min
                
                grid[py0, px0] = v
            
            px = px0
            py = py0
            dx = abs(rx - px)
            dy = -abs(ry - py)
            sx = 1 if px < rx else -1
            sy = 1 if py < ry else -1
            err = dx + dy

            while True:
                if px == rx and py == ry:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    px += sx
                if e2 <= dx:
                    err += dx
                    py += sy

                if not (0 <= px < width and 0 <= py < height):
                    break

                v = grid[py, px] + free_inc
                if v > l_max:
                    v = l_max
                grid[py, px] = v

class MapBuilder:
    def __init__(
        self, 
        resolution: float = 0.05, 
        max_cells: int = 16_000_000,
        occ_dec: float = 0.85,
        free_inc: float = 0.4,
        l_min: float = -4.0,
        l_max: float = 4.0,
        initial_size: int = 512,
        pad_cells: int = 32,
    ) -> None:
        self.resolution = resolution
        self._inv_resolution = 1 / self.resolution
    
        self._occ_dec = occ_dec
        self._free_inc = free_inc
        self._l_min = l_min
        self._l_max = l_max
        self._pad_cells = pad_cells
        self._initial_size = initial_size
        self._grid = np.zeros((initial_size, initial_size), dtype=np.float32) # default size - 1x1 m

        self._max_cells = max_cells
        
        self._min_x = -self._initial_size * self.resolution * 0.5
        self._min_y = -self._initial_size * self.resolution * 0.5
    
    @property
    def grid(self) -> np.ndarray:
        return self._grid
    
    @property
    def min_x(self) -> float:
        return self._min_x
    
    @property
    def min_y(self) -> float:
        return self._min_y
    
    @property
    def width(self) -> int:
        return self._grid.shape[1]
    
    @property
    def height(self) -> int:
        return self._grid.shape[0]
    
    def add_scans(self, scans):
        prepared = self._prepare(scans)
        if prepared is None:
            return

        world_batches, robot_world = prepared
        self._integrate(world_batches, robot_world)
    
    def rebuild(self, keyframes: Iterable[PositionedCloud]):
        self._reset()

        scans = list(keyframes)
        prepared = self._prepare(scans)
        if prepared is None:
            return

        world_batches, robot_world = prepared
        self._integrate(world_batches, robot_world)

    def clear(self) -> None:
        self._grid.fill(0.0)

    def to_occupancy_grid(self, occupied_thresh: float = 0.4) -> OccupancyGrid:
        occ = np.full(self._grid.shape, 127, dtype=np.uint8)
        occ[self.grid < -occupied_thresh] = 0
        occ[self.grid > occupied_thresh] = 255

        return OccupancyGrid(
            occ,
            -int(self._min_x / self.resolution),
            -int(self._min_y / self.resolution),
            self.width,
            self.height,
            self.resolution
        )

    def _reset(self) -> None:
        self._grid = np.zeros(
            (self._initial_size, self._initial_size), dtype=np.float32
        )

        self._min_x = -self._initial_size * self.resolution * 0.5
        self._min_y = -self._initial_size * self.resolution * 0.5

    @staticmethod
    def _prepare(scans: list[PositionedCloud]):
        world_batches = []
        robots = []

        for scan in scans:
            if scan.estimated_pos is None:
                continue

            x, y, th = scan.estimated_pos
            c, s = np.cos(th), np.sin(th)

            pts2 = np.ascontiguousarray(scan.cloud[:, :2], dtype=np.float64)
            world = pts2 @ np.array([[c, s], [-s, c]]) + np.array([x, y])

            world_batches.append(world)
            robots.append((x, y))

        if not world_batches:
            return None

        return world_batches, np.asarray(robots, dtype=np.float64)

    def _integrate(self, world_batches, robot_world):
        all_pts = np.vstack(world_batches + [robot_world])
        ok = self._ensure_bounds(all_pts)

        if not ok:
            max_x = self._min_x + self.width * self.resolution
            max_y = self._min_y + self.height * self.resolution
            clipped = []

            for w in world_batches:
                mask = (
                    (w[:, 0] >= self._min_x) & (w[:, 0] < max_x) &
                    (w[:, 1] >= self._min_y) & (w[:, 1] < max_y)
                )
                clipped.append(w[mask])

            world_batches = clipped

        lengths = np.array([w.shape[0] for w in world_batches], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if offsets[-1] == 0:
            return

        flat = np.empty((int(offsets[-1]), 2), dtype=np.float64)
        for i, w in enumerate(world_batches):
            flat[offsets[i]:offsets[i + 1]] = w

        robot_grid = np.empty((robot_world.shape[0], 2), dtype=np.int64)
        robot_grid[:, 0] = np.floor(
            (robot_world[:, 0] - self._min_x) * self._inv_resolution
        ).astype(np.int64)
        robot_grid[:, 1] = np.floor(
            (robot_world[:, 1] - self._min_y) * self._inv_resolution
        ).astype(np.int64)

        _update_scan_batch(
            self._grid,
            flat,
            offsets,
            robot_grid,
            self._min_x,
            self._min_y,
            self._inv_resolution,
            self.width,
            self.height,
            self._occ_dec,
            self._free_inc,
            self._l_min,
            self._l_max
        )

    def _ensure_bounds(self, world_pts: np.ndarray) -> bool:
        res = self.resolution
        h, w = self._grid.shape

        min_x, min_y = self._min_x, self._min_y
        max_x = min_x + w * res
        max_y = min_y + h * res

        need_min = world_pts.min(axis=0)
        need_max = world_pts.max(axis=0)

        pad = self._pad_cells

        left = max(
            0,
            int(np.ceil((min_x - need_min[0]) * self._inv_resolution)) + pad,
        )
        bottom = max(
            0,
            int(np.ceil((min_y - need_min[1]) * self._inv_resolution)) + pad,
        )
        right = max(
            0,
            int(np.ceil((need_max[0] - max_x) * self._inv_resolution)) + pad,
        )
        top = max(
            0,
            int(np.ceil((need_max[1] - max_y) * self._inv_resolution)) + pad,
        )

        if left == 0 and bottom == 0 and right == 0 and top == 0:
            return True

        new_h = h + bottom + top
        new_w = w + left + right

        if new_h * new_w > self._max_cells:
            return False

        new_grid = np.zeros((new_h, new_w), dtype=self._grid.dtype)
        new_grid[bottom:bottom + h, left:left   + w] = self._grid

        self._grid = new_grid
        self._min_x = min_x - left * res
        self._min_y = min_y - bottom * res

        return True
