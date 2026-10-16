import numpy as np
from collections import deque
import numba

def get_topological_neighbors(
    target_id: int, 
    edges: list, 
    max_steps: int
) -> set[int]:
    adj_list = {}
    for edge in edges:
        u, v = edge.vertex_ids[0], edge.vertex_ids[1]
        
        if u not in adj_list: adj_list[u] = []
        if v not in adj_list: adj_list[v] = []
        
        adj_list[u].append(v)
        adj_list[v].append(u)

    if target_id not in adj_list:
        return set()

    queue = deque([(target_id, 0)])
    visited = {target_id}

    while queue:
        current_id, current_dist = queue.popleft()

        if current_dist >= max_steps:
            continue

        for neighbor in adj_list.get(current_id, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_dist + 1))

    return visited

def evaluate_registration(source, tree, transformation, max_correspondence_distance):
    source_homo = np.hstack([source, np.ones((source.shape[0], 1))])
    source_transformed = (transformation @ source_homo.T).T[:, :3]

    idx, sq_distances = tree.batch_nearest_neighbor_search(
        np.ascontiguousarray(source_transformed, dtype=np.float64)
    )

    sq_distances = np.array(sq_distances)
    idx = np.array(idx)

    inlier_mask = (sq_distances < max_correspondence_distance ** 2) & (idx > -1)
    inlier_sq_distances = sq_distances[inlier_mask]

    if len(source_transformed) > 0:
        fitness = np.sum(inlier_mask) / len(source_transformed)
    else:
        fitness = 0
    
    if len(inlier_sq_distances) > 0:
        inlier_rmse = np.sqrt(np.mean(inlier_sq_distances))
    else:
        inlier_rmse = np.inf

    return fitness, inlier_rmse

@numba.njit(cache=True)
def scan_to_cloud(ranges: np.ndarray, angles: np.ndarray, range_threshold: float = 12.0):    
    valid = (ranges > 0.1) & (ranges < range_threshold)
    ranges = ranges[valid]
    angles = angles[valid]
    
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(angles)

    return np.column_stack((x, y, z))

@numba.njit(cache=True)
def transform_4x4_to_2d_pose(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    
    theta = np.arctan2(matrix[1, 0], matrix[0, 0])
    
    return np.array([x, y, theta])

@numba.njit(cache=True)
def relative_pose(a, b):
    rel_4x4 = relative_pose_4x4(a, b)
    return transform_4x4_to_2d_pose(rel_4x4)

@numba.njit(cache=True)
def relative_pose_4x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    T_w_a = pose2d_to_transform(a)
    T_w_b = pose2d_to_transform(b)
    
    return np.linalg.inv(T_w_a) @ T_w_b

def relative_from_vertices(a, b):
    xa, ya = a.pose.position
    xb, yb = b.pose.position

    ta = a.pose.orientation
    tb = b.pose.orientation

    return relative_pose(np.array([xa, ya, ta]), np.array([xb, yb, tb]))

@numba.njit(cache=True)
def normalize_angle(a):
    while a > np.pi:
        a -= np.pi * 2
        
    while a < -np.pi:
        a += np.pi * 2
        
    return a

@numba.njit(cache=True)
def pose2d_to_transform(pose: np.ndarray) -> np.ndarray:
    x, y, theta = pose
    c = np.cos(theta)
    s = np.sin(theta)
    
    T = np.eye(4)

    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = 0.0
    
    return T
