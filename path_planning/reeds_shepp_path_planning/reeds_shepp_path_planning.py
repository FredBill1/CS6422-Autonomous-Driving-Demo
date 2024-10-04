"""

Reeds Shepp path planner sample code

author Atsushi Sakai(@Atsushi_twi)
co-author Videh Patel(@videh25) : Added the missing RS paths

"""

import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from utils.angle import angle_mod

show_animation = True


@dataclass
class Path:
    lengths: list[float] = field(default_factory=list)  # course segment length (negative value is backward segment)
    ctypes: list[str] = field(default_factory=list)  # course segment type char ("S": straight, "L": left, "R": right)
    L: float = 0.0  # Total lengths of the path
    x: list[float] = field(default_factory=list)  # x positions
    y: list[float] = field(default_factory=list)  # y positions
    yaw: list[float] = field(default_factory=list)  # orientations [rad]
    directions: list[int] = field(default_factory=list)  # directions (1:forward, -1:backward)


def plot_arrow(
    x: float | list[float],
    y: float | list[float],
    yaw: float | list[float],
    length: float = 1.0,
    width: float = 0.5,
    fc="r",
    ec="k",
) -> None:
    if isinstance(x, list):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width
        )
        plt.plot(x, y)


def pi_2_pi(x: float) -> float:
    return angle_mod(x)


def set_path(paths: list[Path], lengths: list[float], ctypes: list[str], step_size: float) -> list[Path]:
    path = Path()
    path.ctypes = ctypes
    path.lengths = lengths
    path.L = sum(np.abs(lengths))

    # check same path exist
    for i_path in paths:
        type_is_same = i_path.ctypes == path.ctypes
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        if type_is_same and length_is_close:
            return paths  # same path found, so do not insert path

    # check path is long enough
    if path.L <= step_size:
        return paths  # too short, so do not insert path

    paths.append(path)
    return paths


def polar(x: float, y: float) -> tuple[float, float]:
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


def left_straight_left(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = pi_2_pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]

    return False, [], []


def left_straight_right(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1**2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = pi_2_pi(t1 + theta)
        v = pi_2_pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ["L", "S", "R"]

    return False, [], []


def left_x_right_x_left(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = pi_2_pi(A + theta + math.pi / 2)
        u = pi_2_pi(math.pi - 2 * A)
        v = pi_2_pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]

    return False, [], []


def left_x_right_left(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = pi_2_pi(A + theta + math.pi / 2)
        u = pi_2_pi(math.pi - 2 * A)
        v = pi_2_pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]

    return False, [], []


def left_right_x_left(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = pi_2_pi(-A + theta + math.pi / 2)
        v = pi_2_pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]

    return False, [], []


def left_right_x_left_right(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = pi_2_pi(theta + A + math.pi / 2)
        u = pi_2_pi(A)
        v = pi_2_pi(phi - t + 2 * u)
        if (t >= 0) and (u >= 0) and (v >= 0):
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]

    return False, [], []


def left_x_right_left_x_right(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if 0 <= u2 <= 1:
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = pi_2_pi(theta + A + math.pi / 2)
        v = pi_2_pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]

    return False, [], []


def left_x_right90_straight_left(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = pi_2_pi(theta + A + math.pi / 2)
        v = pi_2_pi(t - phi + math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "L"]

    return False, [], []


def left_straight_right90_x_left(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = pi_2_pi(theta - A + math.pi / 2)
        v = pi_2_pi(t - phi - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "R", "L"]

    return False, [], []


def left_x_right90_straight_right(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = pi_2_pi(theta + math.pi / 2)
        u = u1 - 2
        v = pi_2_pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "R"]

    return False, [], []


def left_straight_left90_x_right(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = pi_2_pi(theta)
        u = u1 - 2
        v = pi_2_pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "L", "R"]

    return False, [], []


def left_x_right90_straight_left90_x_right(x: float, y: float, phi: float) -> tuple[bool, list[float], list[str]]:
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = pi_2_pi(theta + A + math.pi / 2)
        v = pi_2_pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -math.pi / 2, v], ["L", "R", "S", "L", "R"]

    return False, [], []


def timeflip(travel_distances: list[float]) -> list[float]:
    return [-x for x in travel_distances]


def reflect(steering_directions: list[str]) -> list[str]:
    def switch_dir(dirn):
        if dirn == "L":
            return "R"
        elif dirn == "R":
            return "L"
        else:
            return "S"

    return [switch_dir(dirn) for dirn in steering_directions]


def generate_path(
    q0: tuple[float, float, float], q1: tuple[float, float, float], max_curvature: float, step_size: float
) -> list[Path]:
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    step_size *= max_curvature

    paths = []
    path_functions = [
        left_straight_left,
        left_straight_right,  # CSC
        left_x_right_x_left,
        left_x_right_left,
        left_right_x_left,  # CCC
        left_right_x_left_right,
        left_x_right_left_x_right,  # CCCC
        left_x_right90_straight_left,
        left_x_right90_straight_right,  # CCSC
        left_straight_right90_x_left,
        left_straight_left90_x_right,  # CSCC
        left_x_right90_straight_left90_x_right,
    ]  # CCSCC

    for path_func in path_functions:
        flag, travel_distances, steering_dirns = path_func(x, y, dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, y, -dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(x, -y, -dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, -y, dth)
        if flag:
            for distance in travel_distances:
                if 0.1 * sum([abs(d) for d in travel_distances]) < abs(distance) < step_size:
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths


def calc_interpolate_dists_list(lengths: list[float], step_size: float) -> list[NDArray[np.floating[Any]]]:
    interpolate_dists_list = []
    for length in lengths:
        d_dist = step_size if length >= 0.0 else -step_size
        interp_dists = np.arange(0.0, length, d_dist)
        interp_dists = np.append(interp_dists, length)
        interpolate_dists_list.append(interp_dists)

    return interpolate_dists_list


def generate_local_course(
    lengths: list[float], ctypes: list[str], max_curvature: float, step_size: float
) -> tuple[list[float], list[float], list[float], list[int]]:
    interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size * max_curvature)

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0

    xs, ys, yaws, directions = [], [], [], []
    for interp_dists, ctype, length in zip(interpolate_dists_list, ctypes, lengths):

        for dist in interp_dists:
            x, y, yaw, direction = interpolate(dist, length, ctype, max_curvature, origin_x, origin_y, origin_yaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)
        origin_x = xs[-1]
        origin_y = ys[-1]
        origin_yaw = yaws[-1]

    return xs, ys, yaws, directions


def interpolate(
    dist: float, length: float, ctype: str, max_curvature: float, origin_x: float, origin_y: float, origin_yaw: float
) -> tuple[float, float, float, int]:
    if ctype == "S":
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:  # curve
        ldx = math.sin(dist) / max_curvature
        ldy = 0.0
        yaw = None
        if ctype == "L":  # left turn
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = origin_yaw + dist
        elif ctype == "R":  # right turn
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = origin_yaw - dist
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    return x, y, yaw, 1 if length > 0.0 else -1


def calc_paths(
    start_x: float,
    start_y: float,
    start_yaw: float,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    max_curvature: float,
    step_size: float,
) -> list[Path]:
    q0 = (start_x, start_y, start_yaw)
    q1 = (goal_x, goal_y, goal_yaw)

    paths = generate_path(q0, q1, max_curvature, step_size)
    for path in paths:
        xs, ys, yaws, directions = generate_local_course(path.lengths, path.ctypes, max_curvature, step_size)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for (ix, iy) in zip(xs, ys)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for (ix, iy) in zip(xs, ys)]
        path.yaw = [pi_2_pi(yaw + q0[2]) for yaw in yaws]
        path.directions = directions
        path.lengths = [length / max_curvature for length in path.lengths]
        path.L = path.L / max_curvature

    return paths


def reeds_shepp_path_planning(
    start_x: float,
    start_y: float,
    start_yaw: float,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    max_curvature: float,
    step_size: float = 0.2,
):
    paths = calc_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, max_curvature, step_size)
    if not paths:
        return None, None, None, None, None  # could not generate any path

    # search minimum cost path
    best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
    b_path = paths[best_path_index]

    return b_path.x, b_path.y, b_path.yaw, b_path.ctypes, b_path.lengths


def main():
    print("Reeds Shepp path planner sample start!!")

    start_x = -1.0  # [m]
    start_y = -4.0  # [m]
    start_yaw = np.deg2rad(-20.0)  # [rad]

    end_x = 5.0  # [m]
    end_y = 5.0  # [m]
    end_yaw = np.deg2rad(25.0)  # [rad]

    start_x = 0.0  # [m]
    start_y = 0.0  # [m]
    start_yaw = np.deg2rad(0.0)  # [rad]

    end_x = 0.0  # [m]
    end_y = 1.0  # [m]
    end_yaw = np.deg2rad(0.0)  # [rad]

    curvature = 0.1
    step_size = 0.05

    xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size
    )

    if not xs:
        assert False, "No path"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(xs, ys, label="final course " + str(modes))
        print(f"{lengths=}")

        # plotting
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)

        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    main()
