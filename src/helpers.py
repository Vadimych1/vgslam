import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from collections import deque

def get_topological_neighbors(
    target_id: int, 
    edges: list, 
    max_steps: int
) -> set[int]:
    """
    Returns a set of all vertex IDs within N steps (topological range) 
    of the target_id, excluding the target_id itself.
    """
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

    nx.draw_networkx_nodes(
        G, pos, node_size=100, node_color="crimson", ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, width=1.5, edge_color="navy", alpha=0.7, ax=ax
    )

    label_pos = {node: (coords[0], coords[1] + 0.1) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8, ax=ax)

    ax.set_title("SLAM Graph Layout", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axis("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Graph successfully saved to {filename}")

    plt.close(fig)

def evaluate_registration(source, target, transformation, max_correspondence_distance):
    """
    Evaluate the quality of a point cloud registration.

    Args:
        source (np.ndarray): The source point cloud (Nx3) before transformation.
        target (np.ndarray): The target point cloud (Mx3).
        transformation (np.ndarray): The 4x4 transformation matrix.
        max_correspondence_distance (float): Maximum distance to consider a match.

    Returns:
        tuple: (fitness, inlier_rmse)
            fitness (float): Proportion of source points with a match in the target.
            inlier_rmse (float): RMSE of the inlier correspondences.
    """
    # 1. Transform the source points using the estimated matrix
    # Add a column of ones for homogeneous coordinates
    source_homo = np.hstack([source, np.ones((source.shape[0], 1))])
    source_transformed = (transformation @ source_homo.T).T[:, :3]

    # 2. Build a KD-tree for the target cloud for fast nearest-neighbor search
    tree = KDTree(target)

    # 3. Find the closest point in the target for each transformed source point
    distances, _ = tree.query(source_transformed)

    # 4. Identify inliers (points closer than the threshold)
    inlier_mask = distances < max_correspondence_distance
    inlier_distances = distances[inlier_mask]

    # 5. Calculate the metrics
    fitness = np.sum(inlier_mask) / source.shape[0]
    
    if len(inlier_distances) > 0:
        inlier_rmse = np.sqrt(np.mean(inlier_distances ** 2))
    else:
        inlier_rmse = np.inf  # No inliers found, match is terrible

    return fitness, inlier_rmse

def scan_to_cloud(ranges: list | np.ndarray, angles: list | np.ndarray, range_threshold: float = 12.0):
    ranges = np.array(ranges)
    angles = np.array(angles)
    
    valid = (ranges > 0.1) & (ranges < range_threshold)
    ranges = ranges[valid]
    angles = angles[valid]
    
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(angles)

    return np.column_stack((x, y, z))

def transform_4x4_to_2d_pose(matrix):
    """
    Extracts x, y, and theta (in radians) from a 4x4 homogeneous transformation matrix.
    """
    x = matrix[0, 3]
    y = matrix[1, 3]
    
    theta = np.arctan2(matrix[1, 0], matrix[0, 0])
    
    return np.array([x, y, theta])

def relative_pose(a, b):
    xa, ya, ta = a
    xb, yb, tb = b

    dx_w = xb - xa
    dy_w = yb - ya

    c = np.cos(ta)
    s = np.sin(ta)

    dx = c * dx_w + s * dy_w
    dy = -s * dx_w + c * dy_w

    dtheta = np.arctan2(
        np.sin(tb - ta),
        np.cos(tb - ta)
    )

    return [dx, dy, dtheta]

def invert_pose(p):
    x, y, theta = p

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        -c * x - s * y,
         s * x - c * y,
        -theta
    ])

def relative_from_vertices(a, b):
    xa, ya = a.pose.position
    xb, yb = b.pose.position

    ta = a.pose.orientation
    tb = b.pose.orientation

    dx_w = xb - xa
    dy_w = yb - ya

    c = np.cos(ta)
    s = np.sin(ta)

    dx = c * dx_w + s * dy_w
    dy = -s * dx_w + c * dy_w

    dt = np.arctan2(
        np.sin(tb - ta),
        np.cos(tb - ta)
    )

    return np.array([dx, dy, dt])
