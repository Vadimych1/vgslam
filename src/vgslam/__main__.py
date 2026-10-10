from . import _RobotSimulator, VGSLAM, PositionedCloud, scan_to_cloud
from .src.helpers import save_slam_graph, plot_points

import numpy as np
import cv2 as cv

sim = _RobotSimulator()
slam = VGSLAM()

scans = [sim.step(speed=0.5) for _ in range(300)]

poses_slam = []
poses_real = []
poses_odometry = []
for i, scan in enumerate(scans):
    pos, closed = slam.process_scan(
        PositionedCloud(
            scan_to_cloud(scan.ranges, scan.angles, 12),
            scan.odom_pos,
        )
    )
    
    if pos is not None:
        poses_slam.append(pos)
        poses_odometry.append(scan.odom_pos)
        poses_real.append(scan.true_pos)

slam.save_to_file("abc.bin")

slam.optimize()
poses_final = [v.pose.position for v in slam.vertices.values()]

save_slam_graph(list(slam.vertices.values()), slam.edges)
plot_points(np.array(poses_slam), np.array(poses_real), np.array(poses_odometry), np.array(poses_final))
cv.imwrite("map.png", slam.create_occupancy_grid().grid)
