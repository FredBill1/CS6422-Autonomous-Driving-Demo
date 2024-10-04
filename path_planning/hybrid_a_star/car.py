"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

from collections.abc import Iterable
from math import cos, pi, sin, tan

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from utils.angle import rot_mat_2d

WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def check_car_collision(
    x_list: list[float],
    y_list: list[float],
    yaw_list: list[float],
    obstacle_x_list: list[float],
    obstacle_y_list: list[float],
    kd_tree: cKDTree,
) -> bool:
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw, [obstacle_x_list[i] for i in ids], [obstacle_y_list[i] for i in ids]):
            return False  # collision

    return True  # no collision


def rectangle_check(x: float, y: float, yaw: float, obstacle_x_list: list[float], obstacle_y_list: list[float]) -> bool:
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(obstacle_x_list, obstacle_y_list):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # collision

    return True  # no collision


def plot_arrow(
    x: float | Iterable[float],
    y: float | Iterable[float],
    yaw: float | Iterable[float],
    length: float = 1.0,
    width: float = 0.5,
    fc="r",
    ec="k",
) -> None:
    """Plot arrow."""
    if not isinstance(x, float):
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(
            x, y, length * cos(yaw), length * sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4
        )


def plot_car(x: float, y: float, yaw: float) -> None:
    car_color = "-k"
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def move(x: float, y: float, yaw: float, distance: float, steer: float, L: float = WB) -> tuple[float, float, float]:
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return x, y, yaw


def main():
    x, y, yaw = 0.0, 0.0, 1.0
    plt.axis("equal")
    plot_car(x, y, yaw)
    plt.show()


if __name__ == "__main__":
    main()
