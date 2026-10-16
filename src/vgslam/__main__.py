# Vadimych1, 2026
# https://github.com/Vadimych1
#
# example of using VGSLAM
#
# outputs these files:
# - data.vslm      - saved SLAM state
# - output.mp4     - record of map changing in realtime
# - map.png        - final produced grid map
# - pose_graph.png - real, odometry-based and estimated trajectories plotted
# - slam_graph.png - internal pose graph showing loop closures

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv

from . import _RobotSimulator, VGSLAM, PositionedCloud, scan_to_cloud

def plot_points(poses_1, poses_2, poses_3, final):
    x1, y1 = poses_1[:, 0], poses_1[:, 1]
    x2, y2 = poses_2[:, 0], poses_2[:, 1]
    x3, y3 = poses_3[:, 0], poses_3[:, 1]
    xf, yf = final[:, 0], final[:, 1]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x1, y1, color='blue', linestyle=':', marker='o', alpha=0.3, label='VGSLAM-Estimated')
    plt.plot(x2, y2, color='red', linestyle=':', marker='s', alpha=0.3, label='Ground Truth')
    plt.plot(x3, y3, color='green', linestyle=':', marker='D', alpha=0.3, label='Odometry')
    plt.plot(xf, yf, color='yellow', linestyle='-', marker='*', alpha=0.5, label='Final')

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
    

# robot with 2d-lidar and odometry simulation
sim = _RobotSimulator(
    odom_rotation_noise=np.deg2rad(0.05),
    odom_bias_rotation_walk=np.deg2rad(0.05),
    odom_bias_translation_walk=0.02
)

# VGSLAM initialization
slam = VGSLAM(
    loop_closure_edge_uncertainty=0.2,
    odometry_edge_uncertainty=1.0
)

# pre-generated scans
scans = [sim.step(speed=0.5) for _ in range(300)]

res = 0.05
view_width = 20
view_height = 10

w = int(round(view_width / res))
h = int(round(view_height / res))
w = w + (w % 2)
h = h + (h % 2)

cx = w // 2
cy = h // w

# video writer for map recording
fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer = cv.VideoWriter(
    "output.mp4",
    fourcc,
    10,
    (w, h), isColor=True
)

# arrays for saving real, odometry and estimated poses
poses_slam = []
poses_real = []
poses_odometry = []

print("[] starting")
for i, scan in enumerate(scans):
    # main VGSLAM function - processes scan to estimate current position
    # this example uses scan_to_cloud(ranges, angles, threshold) 
    # to convert LiDAR scan to point cloud
    pos, closed = slam.process_scan(
        PositionedCloud(
            scan_to_cloud(scan.ranges, scan.angles, 12),
            scan.odom_pos,
        )
    )
    
    # when robot reaches ~middle of its way
    # we are testing map saving and probabilistic position restoration
    if i == 150:
        # outputs which vertex was created last so you could see
        # the graph change after new session creation
        print("[] graph cut at", slam._graph.last_vertex_id)
        
        # save all the data to one file
        slam.save_to_file("data.vslm")
        print("[] saved state to file")
        
        # and load it then
        slam = VGSLAM.load_from_file("data.vslm")
        
        # reset position and estimation cache
        #
        # by providing None instead of initial position we are allowing the algorithm
        # to find itself on the map using probabilistic search
        slam.new_session(None)
        
        print("[] loaded state from file & started new session")
    
    # save pose
    if pos is not None:
        poses_slam.append(pos)
        poses_odometry.append(scan.odom_pos)
        poses_real.append(scan.true_pos)

    # and do some magic to make a video
    canvas = np.full((h, w), 127, dtype=np.uint8)
    
    occ = slam.create_occupancy_grid()
    grid = occ.grid
    zx, zy = occ.zero_offset_x_px, occ.zero_offset_y_px
    cres = occ.resolution
    
    gh, gw = grid.shape
    
    ox = int(round(cx - zx))
    oy = int(round(cy - zy)) + int(5 / res)
    
    src_x0 = max(0, -ox)
    src_y0 = max(0, -oy)
    src_x1 = min(gw, w - ox)
    src_y1 = min(gh, h - oy)
    
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[
            oy + src_y0 : oy + src_y1,
            ox + src_x0 : ox + src_x1,
        ] = grid[src_y0:src_y1, src_x0:src_x1]
    
    frame = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
    writer.write(frame)

print("[] done")

# do graph final optimization
# (usually it is done when loop closure happens)
slam._graph.optimize()

# plot graph and all saved poses
poses_final = [v.pose.position for v in slam._graph.vertices.values()]

save_slam_graph(list(slam._graph.vertices.values()), slam._graph.edges)
plot_points(np.array(poses_slam), np.array(poses_real), np.array(poses_odometry), np.array(poses_final))

writer.release()

# calculate RMSE
poses_real = np.array(poses_real)
poses_slam = np.array(poses_slam)

trans1, rot1 = poses_real[:, :2], poses_real[:, 2]
trans2, rot2 = poses_slam[:, :2], poses_slam[:, 2]

trans_diff = trans1 - trans2
trans_sq_dist = np.sum(trans_diff ** 2, axis=1)
trans_rmse = np.sqrt(np.mean(trans_sq_dist))

rot_diff = rot1 - rot2
rot_diff_wrapped = (rot_diff + np.pi) % (2 * np.pi) - np.pi
rot_rmse = np.sqrt(np.mean(rot_diff_wrapped ** 2))

print("pos rmse (real to slam):", trans_rmse)
print("ang rmse (real to slam):", rot_rmse)
