import numpy as np
from dataclasses import dataclass


@dataclass
class SimScan:
    timestamp: float
    true_pos: np.ndarray
    odom_pos: np.ndarray
    cloud: np.ndarray
    angles: np.ndarray
    ranges: np.ndarray


class RobotSimulator:
    def __init__(
        self,
        dt=0.1,
        lidar_angle_step=np.deg2rad(1.0),
        lidar_max_range=10.0,
        lidar_noise_std=0.005,

        # odometry noise
        odom_translation_noise=0.003,
        odom_rotation_noise=np.deg2rad(0.01),

        # slowly varying odometry bias
        odom_bias_translation_walk=0.0005,
        odom_bias_rotation_walk=np.deg2rad(0.01),
    ):
        self.dt = dt

        # rectangle room constraints
        self.x_min = -5.0
        self.x_max = 5.0
        self.y_min = -2.0
        self.y_max = 2.0

        self.angles = np.arange(
            -np.pi,
            np.pi,
            lidar_angle_step
        )

        self.lidar_max_range = lidar_max_range
        self.lidar_noise_std = lidar_noise_std

        self.odom_translation_noise = odom_translation_noise
        self.odom_rotation_noise = odom_rotation_noise

        self.odom_bias_translation_walk = (
            odom_bias_translation_walk
        )

        self.odom_bias_rotation_walk = (
            odom_bias_rotation_walk
        )

        self.odom_bias_xy = np.zeros(2)
        self.odom_bias_theta = 0.0

        self.true_pos = np.array(
            [0.0, 0.0, 0.0],
            dtype=float
        )

        self.odom_pos = np.array(
            [0.0, 0.0, 0.0],
            dtype=float
        )

        self.prev_true_pos = self.true_pos.copy()

        # path
        self.waypoints = np.array([
            [0.0, 0.0],
            [4.0, 0.0],
            [1.0, 0.5],
            [1.0, -0.5],
            [0.0, -0.8],
            [0.0, 0.8],
            [4.0, 0.8],
        ])

        self.segment = 0
        self.segment_progress = 0.0

        self.timestamp = 0.0

    def _move_along_trajectory(self, speed=0.5):
        distance_to_move = speed * self.dt

        while distance_to_move > 0:

            if self.segment >= len(self.waypoints) - 1:
                return

            p0 = self.waypoints[self.segment]
            p1 = self.waypoints[self.segment + 1]

            segment_vec = p1 - p0
            segment_length = np.linalg.norm(segment_vec)

            direction = segment_vec / segment_length

            remaining = (
                segment_length
                - self.segment_progress
            )

            move = min(
                distance_to_move,
                remaining
            )

            self.segment_progress += move
            distance_to_move -= move

            position = (
                p0
                + direction * self.segment_progress
            )

            theta = np.arctan2(
                direction[1],
                direction[0]
            )

            self.true_pos = np.array([
                position[0],
                position[1],
                theta
            ])

            if self.segment_progress >= segment_length:
                self.segment += 1
                self.segment_progress = 0.0

    def _update_odometry(self, dt, speed):
        true_prev = self.prev_true_pos
        true_now = self.true_pos

        dx_world = (
            true_now[0] - true_prev[0]
        )

        dy_world = (
            true_now[1] - true_prev[1]
        )

        c = np.cos(true_prev[2])
        s = np.sin(true_prev[2])

        dx = (
            c * dx_world
            + s * dy_world
        )

        dy = (
            -s * dx_world
            + c * dy_world
        )

        dtheta = self._angle_diff(
            true_now[2],
            true_prev[2]
        )

        self.odom_bias_xy += np.random.normal(
            0.0,
            self.odom_bias_translation_walk,
            2
        ) * np.sqrt(dt) * speed

        self.odom_bias_theta += np.random.normal(
            0.0,
            self.odom_bias_rotation_walk
        ) * np.sqrt(dt) * speed

        dx_noisy = (
            dx
            + self.odom_bias_xy[0]
            + np.random.normal(
                0.0,
                self.odom_translation_noise
            ) * dt * speed
        )

        dy_noisy = (
            dy
            + self.odom_bias_xy[0]
            + np.random.normal(
                0.0,
                self.odom_translation_noise
            ) * dt * speed
        )

        dtheta_noisy = (
            dtheta
            + self.odom_bias_theta
            + np.random.normal(
                0.0,
                self.odom_rotation_noise
            ) * dt * speed
        )

        c = np.cos(self.odom_pos[2])
        s = np.sin(self.odom_pos[2])

        self.odom_pos[0] += (
            dx_noisy * c
            - dy_noisy * s
        )

        self.odom_pos[1] += (
            dx_noisy * s
            + dy_noisy * c
        )

        self.odom_pos[2] = self._normalize_angle(
            self.odom_pos[2] + dtheta_noisy
        )

        self.prev_true_pos = true_now.copy()

    def _lidar_scan(self):
        x, y, theta = self.true_pos

        ranges = np.full(
            len(self.angles),
            self.lidar_max_range,
            dtype=float
        )

        # Ray direction in world frame
        ray_angles = self.angles + theta

        dx = np.cos(ray_angles)
        dy = np.sin(ray_angles)

        for i in range(len(self.angles)):

            vx = dx[i]
            vy = dy[i]

            candidates = []

            # -------------------------------------------------
            # Vertical wall: x = x_min
            # -------------------------------------------------

            if abs(vx) > 1e-12:

                t = (self.x_min - x) / vx

                if t > 0:

                    yy = y + t * vy

                    if self.y_min <= yy <= self.y_max:
                        candidates.append(t)

            # x = x_max
            if abs(vx) > 1e-12:

                t = (self.x_max - x) / vx

                if t > 0:

                    yy = y + t * vy

                    if self.y_min <= yy <= self.y_max:
                        candidates.append(t)

            # -------------------------------------------------
            # Horizontal wall: y = y_min
            # -------------------------------------------------

            if abs(vy) > 1e-12:

                t = (self.y_min - y) / vy

                if t > 0:

                    xx = x + t * vx

                    if self.x_min <= xx <= self.x_max:
                        candidates.append(t)

            # y = y_max
            if abs(vy) > 1e-12:

                t = (self.y_max - y) / vy

                if t > 0:

                    xx = x + t * vx

                    if self.x_min <= xx <= self.x_max:
                        candidates.append(t)

            if candidates:
                ranges[i] = min(candidates)

        # -------------------------------------------------
        # Add LiDAR measurement noise
        # -------------------------------------------------

        ranges += np.random.normal(
            0.0,
            self.lidar_noise_std,
            len(ranges)
        )

        ranges = np.clip(
            ranges,
            0.0,
            self.lidar_max_range
        )

        # -------------------------------------------------
        # Convert polar → LOCAL Cartesian coordinates
        # -------------------------------------------------

        local_x = ranges * np.cos(self.angles)
        local_y = ranges * np.sin(self.angles)

        cloud = np.column_stack([
            local_x,
            local_y,
            np.zeros(len(ranges))
        ])

        return ranges, cloud

    def step(self, speed=0.5):
        self._move_along_trajectory(speed)
        self._update_odometry(self.dt, speed)

        ranges, cloud = self._lidar_scan()

        scan = SimScan(
            timestamp=self.timestamp,
            true_pos=self.true_pos.copy(),
            odom_pos=self.odom_pos.copy(),
            cloud=cloud,
            angles=self.angles.copy(),
            ranges=ranges,
        )

        self.timestamp += self.dt

        return scan

    @staticmethod
    def _normalize_angle(angle):
        return (
            angle + np.pi
        ) % (2 * np.pi) - np.pi

    @staticmethod
    def _angle_diff(a, b):
        return RobotSimulator._normalize_angle(a - b)
