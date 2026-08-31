import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import numba

@numba.njit(cache=True)
def world_to_grid(points: np.ndarray, min_x: float, min_y: float, resolution: float) -> np.ndarray:
    g = np.floor((points - np.array([min_x, min_y])) / resolution)
    return g.astype(np.int64)

@numba.njit(cache=True)
def update_scan(
    grid,
    perscan_points,
    robot_positions,
    min_x,
    min_y,
    resolution,
    width,
    height,
    occ_dec,
    free_inc,
):
    for scan_idx in range(len(perscan_points)):
        rx = robot_positions[scan_idx, 0]
        ry = robot_positions[scan_idx, 1]
        w_points = perscan_points[scan_idx]
        
        points = np.floor((w_points - np.array([min_x, min_y])) / resolution).astype(np.int64)
        
        for i in range(points.shape[0]):
            px0 = points[i, 0]
            py0 = points[i, 1]

            if 0 <= px0 < width and 0 <= py0 < height:
                grid[py0, px0] -= occ_dec

            px, py = px0, py0
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
                    
                if 0 <= px < width and 0 <= py < height:
                    grid[py, px] += free_inc
    
    np.clip(grid, -4, 4, out=grid)

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

    # distances, _ = tree.query(source_transformed)
    # inlier_mask = distances < max_correspondence_distance
    # inlier_distances = distances[inlier_mask]

    idx, sq_distances = tree.batch_nearest_neighbor_search(
        np.ascontiguousarray(source_transformed, dtype=np.float64)
    )

    sq_distances = np.array(sq_distances)
    idx = np.array(idx)

    inlier_mask = (sq_distances < max_correspondence_distance ** 2) & (idx > -1)
    inlier_sq_distances = sq_distances[inlier_mask]

    fitness = np.sum(inlier_mask) / len(source_transformed)
    
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

def plot_points(poses_1, poses_2, poses_3, final):
    x1, y1 = poses_1[:, 0], poses_1[:, 1]
    x2, y2 = poses_2[:, 0], poses_2[:, 1]
    x3, y3 = poses_3[:, 0], poses_3[:, 1]
    xf, yf = final[:, 0], final[:, 1]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x1, y1, color='blue', linestyle=':', marker='o', alpha=0.3, label='Path 1 Trajectory')
    plt.plot(x2, y2, color='red', linestyle=':', marker='s', alpha=0.3, label='Path 2 Trajectory')
    plt.plot(x3, y3, color='green', linestyle=':', marker='D', alpha=0.3, label='Path 3 Trajectory')
    plt.plot(xf, yf, color='yellow', linestyle='-', marker='*', alpha=0.5, label='Final Trajectory')

    plt.title('Pose Graph Trajectory', fontsize=14, fontweight='bold')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')

    plt.savefig('pose_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_slam_graph(
    vertices: list, edges: list, filename: str = "slam_graph.png"
):
    G = nx.Graph()

    pos = {}
    for vertex in vertices:
        v_id = vertex.id
        x, y = vertex.pose.position
        pos[v_id] = (x, y)
        G.add_node(v_id)

    for edge in edges:
        if len(edge.vertex_ids) == 2:
            G.add_edge(edge.vertex_ids[0], edge.vertex_ids[1])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_on()
    ax.tick_params(
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
    )

    nx.draw_networkx_nodes(
        G, pos, node_size=100, node_color="crimson", ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, width=1.5, edge_color="navy", alpha=0.7, ax=ax
    )

    label_pos = {node: (coords[0], coords[1] + 0.1) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8, ax=ax)

    ax.set_title('Pose Graph Trajectory', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
