from dataclasses import dataclass, replace

import numpy as np

from ..utils.wrap_angle import wrap_angle
from .Obstacles import Obstacles


@dataclass(slots=True)
class Car:
    x: float  # [m]
    y: float  # [m]
    yaw: float  # [rad], [-pi, pi)
    velocity: float = 0.0  # [m/s], [MIN_SPEED, MAX_SPEED]
    steer: float = 0.0  # [rad], [-MAX_STEER, MAX_STEER]

    WHEEL_BASE = 2.5  # [m]

    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACK_TO_WHEEL = 1.0  # [m]
    WHEEL_LENGTH = 0.8  # [m]
    WHEEL_WIDTH = 0.5  # [m]
    WHEEL_SPACING = 1.4  # [m]

    BACK_TO_CENTER = LENGTH / 2 - BACK_TO_WHEEL  # [m]

    COLLISION_LENGTH = LENGTH + 1.0  # [m]
    COLLISION_WIDTH = WIDTH + 1.0  # [m]
    COLLISION_RADIUS = np.hypot(COLLISION_WIDTH / 2, COLLISION_LENGTH / 2)  # [m]

    MAX_STEER = np.deg2rad(40.0)  # [rad]
    TARGET_MAX_STEER = np.deg2rad(35.0)  # [rad], for global planner
    MAX_STEER_SPEED = np.deg2rad(360.0)  # [rad/s]
    MAX_SPEED = 55.0 / 3.6  #  [m/s]
    MIN_SPEED = -30.0 / 3.6  #  [m/s]
    MAX_ACCEL = 15.0  #  [m/ss]

    TARGET_SPEED = 40.0 / 3.6  # [m/s]

    TARGET_MIN_TURNING_RADIUS = WHEEL_BASE / np.tan(TARGET_MAX_STEER)  # [m], for global planner

    SCAN_RADIUS = 15.0  # [m]

    def align_yaw(self, target_yaw: float) -> None:
        "align the car's yaw to the target yaw, ensuring the angular distance is less than pi"
        self.yaw = target_yaw + wrap_angle(self.yaw - target_yaw)

    def update(self, dt: float, *, do_wrap_angle: bool = True) -> None:
        "predict the next state of the car after `dt` seconds"
        v, s = self.velocity, self.steer
        self.x += v * np.cos(self.yaw) * dt
        self.y += v * np.sin(self.yaw) * dt
        self.yaw += v / self.WHEEL_BASE * np.tan(s) * dt
        if do_wrap_angle:
            self.yaw = wrap_angle(self.yaw)

    def update_with_control(
        self,
        target_velocity: float,
        target_steer: float,
        dt: float,
        *,
        do_wrap_angle: bool = True,
    ) -> None:
        "predict the next state of the car after `dt` seconds with the given control inputs"
        self.update(dt, do_wrap_angle=do_wrap_angle)
        # clip by the maximum values
        target_velocity = np.clip(target_velocity, self.MIN_SPEED, self.MAX_SPEED)
        target_steer = np.clip(target_steer, -self.MAX_STEER, self.MAX_STEER)
        # clip by the maximum accels
        self.velocity += np.clip(target_velocity - self.velocity, -self.MAX_ACCEL * dt, self.MAX_ACCEL * dt)
        self.steer += np.clip(target_steer - self.steer, -self.MAX_STEER_SPEED * dt, self.MAX_STEER_SPEED * dt)

    def copy(self) -> "Car":
        return replace(self)

    def check_collision(self, obstacles: Obstacles | np.ndarray) -> bool:
        "Check if the car collides with any obstacles in the given `Obstacles` instance."

        # calculate the center of the car, since (self.x, self.y) represents the coordinate of the middle of the rear wheels
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        center_x, center_y = self.x + self.BACK_TO_CENTER * c, self.y + self.BACK_TO_CENTER * s

        if isinstance(obstacles, Obstacles):
            # query the obstacles within the collision radius
            ids = obstacles.kd_tree.query_ball_point([center_x, center_y], self.COLLISION_RADIUS)
            candidates = obstacles.coordinates[ids]
        else:
            # the input is already the coordinates of the obstacles
            candidates = obstacles

        # translate and then rotate the coordinates of the obstacles to the car's local frame, to facilitate checking
        candidates = (candidates - [center_x, center_y]) @ np.array([[c, -s], [s, c]])

        return np.any(
            np.logical_and(
                np.abs(candidates[:, 0]) < self.COLLISION_LENGTH / 2,
                np.abs(candidates[:, 1]) < self.COLLISION_WIDTH / 2,
            )
        )
