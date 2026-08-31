from . import _RobotSimulator, VGSLAM, PositionedCloud, scan_to_cloud
from .src.helpers import save_slam_graph, plot_points

import numpy as np
import cv2 as cv

sim = _RobotSimulator(
    dt=0.1,
    lidar_angle_step=np.deg2rad(0.5),
    odom_rotation_noise=np.deg2rad(0.1),
    odom_translation_noise=0.03,
    odom_bias_translation_walk=0.002,
    odom_bias_rotation_walk=np.deg2rad(0.05)
)
slam = VGSLAM()

scans = [
    sim.step(speed=0.2) for _ in range(700)
]

poses_slam = []
poses_real = []
poses_odometry = []
for scan in scans:
    pos, closed = slam.process_scan(
        PositionedCloud(
            scan_to_cloud(scan.ranges, scan.angles, 12),
            scan.odom_pos * np.array([1, 1, -1]),
        )
    )
    
    if pos is not None:
        poses_slam.append(pos)
        poses_odometry.append(scan.odom_pos)
        poses_real.append(scan.true_pos)

slam.optimize()
poses_final = [v.pose.position for v in slam.vertices.values()]

save_slam_graph(list(slam.vertices.values()), slam.edges)
plot_points(np.array(poses_slam), np.array(poses_real), np.array(poses_odometry), np.array(poses_final))
cv.imwrite("map.png", slam.create_occupancy_grid())
