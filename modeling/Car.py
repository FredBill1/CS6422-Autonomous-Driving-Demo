from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np

from utils.wrap_angle import wrap_angle

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

    COLLISION_LENGTH = LENGTH + 2.0  # [m]
    COLLISION_WIDTH = WIDTH + 2.0  # [m]
    COLLISION_RADIUS = np.hypot(COLLISION_WIDTH / 2, COLLISION_LENGTH / 2)  # [m]

    MAX_STEER = np.deg2rad(40.0)  # [rad]
    TARGET_MAX_STEER = np.deg2rad(35.0)  # [rad], for global planner
    MAX_STEER_SPEED = np.deg2rad(360.0)  # [rad/s]
    MAX_SPEED = 55.0 / 3.6  #  [m/s]
    MIN_SPEED = -20.0 / 3.6  #  [m/s]
    MAX_ACCEL = 15.0  #  [m/ss]

    TARGET_SPEED = 40.0 / 3.6  # [m/s]

    TARGET_MIN_TURNING_RADIUS = WHEEL_BASE / np.tan(TARGET_MAX_STEER)  # [m], for global planner

    def update(self, dt: float, /, do_wrap_angle: bool = False) -> None:
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += self.velocity / self.WHEEL_BASE * np.tan(self.steer) * dt
        if do_wrap_angle:
            self.yaw = wrap_angle(self.yaw)

    def update_with_control(
        self, target_velocity: float, target_steer: float, dt: float, /, do_wrap_angle: bool = False
    ) -> None:
        self.update(dt, do_wrap_angle=do_wrap_angle)
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
