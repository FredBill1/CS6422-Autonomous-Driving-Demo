from dataclasses import dataclass, replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wrap_angle import wrap_angle

from .Obstacles import Obstacles


@dataclass(slots=True)
class Car:
    x: float | npt.NDArray[np.floating[Any]]  # [m]
    y: float | npt.NDArray[np.floating[Any]]  # [m]
    yaw: float | npt.NDArray[np.floating[Any]]  # [rad], [-pi, pi)
    velocity: float | npt.NDArray[np.floating[Any]] = 0.0  # [m/s], [MIN_SPEED, MAX_SPEED]
    steer: float | npt.NDArray[np.floating[Any]] = 0.0  # [rad], [-MAX_STEER, MAX_STEER]

    WHEEL_BASE = 2.5  # [m]

    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACK_TO_WHEEL = 1.0  # [m]
    WHEEL_LENGTH = 0.8  # [m]
    WHEEL_WIDTH = 0.5  # [m]
    WHEEL_SPACING = 1.4  # [m]

    BACK_TO_CENTER = LENGTH / 2 - BACK_TO_WHEEL  # [m]

    COLLISION_LENGTH = LENGTH + 2.0  # [m]
    COLLISION_WIDTH = WIDTH + 2.0  # [m]
    COLLISION_RADIUS = np.hypot(COLLISION_WIDTH / 2, COLLISION_LENGTH / 2)  # [m]

    MAX_STEER = np.deg2rad(40.0)  # [rad]
    TARGET_MAX_STEER = np.deg2rad(35.0)  # [rad], for global planner
    MAX_STEER_SPEED = np.deg2rad(360.0)  # [rad/s]
    MAX_SPEED = 55.0 / 3.6  #  [m/s]
    MIN_SPEED = -30.0 / 3.6  #  [m/s]
    MAX_ACCEL = 15.0  #  [m/ss]

    TARGET_SPEED = 40.0 / 3.6  # [m/s]

    TARGET_MIN_TURNING_RADIUS = WHEEL_BASE / np.tan(TARGET_MAX_STEER)  # [m], for global planner

    LIDAR_RANGE = 20.0  # [m]
    LIDAR_SIGMA = 1.0  # [m]
    CONTROL_SIGMA = [0.1, np.deg2rad(3.0)]  # [m/s, rad], [velocity, steer]

    def align_yaw(self, target_yaw: float) -> None:
        self.yaw = target_yaw + wrap_angle(self.yaw - target_yaw)

    def update(self, dt: float, /, do_wrap_angle: bool = True, with_noise: bool = False) -> None:
        v, s = self.velocity, self.steer
        if with_noise:
            v = np.random.randn(*np.shape(self.x)) * self.CONTROL_SIGMA[0] + v
            s = np.random.randn(*np.shape(self.x)) * self.CONTROL_SIGMA[1] + s
        self.x += v * np.cos(self.yaw) * dt
        self.y += v * np.sin(self.yaw) * dt
        self.yaw += v / self.WHEEL_BASE * np.tan(s) * dt
        if do_wrap_angle:
            yaw = wrap_angle(self.yaw)
            if isinstance(self.yaw, np.ndarray):
                self.yaw[:] = yaw
            else:
                self.yaw = yaw

    def update_with_control(
        self,
        target_velocity: float,
        target_steer: float,
        dt: float,
        /,
        do_wrap_angle: bool = True,
        with_noise: bool = False,
    ) -> None:
        self.update(dt, do_wrap_angle=do_wrap_angle, with_noise=with_noise)
        target_velocity = np.clip(target_velocity, self.MIN_SPEED, self.MAX_SPEED)
        target_steer = np.clip(target_steer, -self.MAX_STEER, self.MAX_STEER)
        self.velocity += np.clip(target_velocity - self.velocity, -self.MAX_ACCEL * dt, self.MAX_ACCEL * dt)
        self.steer += np.clip(target_steer - self.steer, -self.MAX_STEER_SPEED * dt, self.MAX_STEER_SPEED * dt)

    def copy(self) -> "Car":
        return replace(self)

    def check_collision(self, obstacles: Obstacles) -> bool:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        center_x, center_y = self.x + self.BACK_TO_CENTER * c, self.y + self.BACK_TO_CENTER * s
        ids = obstacles.kd_tree.query_ball_point([center_x, center_y], self.COLLISION_RADIUS)
        candidates = obstacles.coordinates[ids]
        candidates = (candidates - [center_x, center_y]) @ np.array([[c, s], [-s, c]])
        return np.any(
            np.logical_and(
                np.abs(candidates[:, 0]) < self.COLLISION_LENGTH / 2,
                np.abs(candidates[:, 1]) < self.COLLISION_WIDTH / 2,
            )
        )

    def plot(self, ax: plt.Axes, color="-k") -> list[plt.Artist]:
        BOX = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]]) / 2
        outline = BOX * [self.LENGTH, self.WIDTH] + [self.LENGTH / 2 - self.BACK_TO_WHEEL, 0]
        wheel = BOX * [self.WHEEL_LENGTH, self.WHEEL_WIDTH]

        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cs, ss = np.cos(self.steer), np.sin(self.steer)
        rot1 = np.array([[cy, -sy], [sy, cy]])
        rot2 = np.array([[cs, -ss], [ss, cs]])
        f_wheel = (rot2 @ wheel.T).T
        fl_wheel = f_wheel + [self.WHEEL_BASE, self.WHEEL_SPACING / 2]
        fr_wheel = f_wheel + [self.WHEEL_BASE, -self.WHEEL_SPACING / 2]
        rl_wheel = wheel + [0, self.WHEEL_SPACING / 2]
        rr_wheel = wheel + [0, -self.WHEEL_SPACING / 2]

        artists = []
        for box in (outline, fl_wheel, fr_wheel, rl_wheel, rr_wheel):
            box = (rot1 @ box.T).T + [self.x, self.y]
            artists.extend(ax.plot(*box.T, color))
        artists.extend(ax.plot(self.x, self.y, "*"))
        return artists

    def lidar_scan(self, obstacles: Obstacles, /, with_noise: bool = False) -> npt.NDArray[np.floating[Any]]:
        ids = obstacles.kd_tree.query_ball_point([self.x, self.y], self.LIDAR_RANGE)
        scan = obstacles.coordinates[ids]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        scan = (scan - [self.x, self.y]) @ np.array([[c, s], [-s, c]])
        if with_noise:
            scan += np.random.randn(*scan.shape) * self.LIDAR_SIGMA
        return scan
