"""

A* grid based planning

author: Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import heapq
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt

show_animation = False


@dataclass
class Node:
    x: int
    y: int
    cost: float
    parent_index: int


def calc_distance_heuristic(
    goal_x: float,
    goal_y: float,
    obstacle_x_list: list[float],
    obstacle_y_list: list[float],
    resolution: float,
    rr: float,
) -> dict[int, Node]:
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    resolution: grid resolution [m]
    rr: robot radius[m]
    """

    goal_node = Node(round(goal_x / resolution), round(goal_y / resolution), 0.0, -1)
    obstacle_x_list = [iox / resolution for iox in obstacle_x_list]
    obstacle_y_list = [ioy / resolution for ioy in obstacle_y_list]

    obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w = _calc_obstacle_map(
        obstacle_x_list, obstacle_y_list, resolution, rr
    )

    motion = _get_motion_model()

    open_set: dict[int, Node] = {}
    closed_set: dict[int, Node] = {}
    open_set[_calc_index(goal_node, x_w, min_x, min_y)] = goal_node
    priority_queue: list[tuple[float, int]] = [(0.0, _calc_index(goal_node, x_w, min_x, min_y))]

    while True:
        if not priority_queue:
            break
        cost, c_id = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        # show graph
        if show_animation:  # pragma: no cover
            plt.plot(current.x * resolution, current.y * resolution, "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event", lambda event: [exit(0) if event.key == "escape" else None]
            )
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.1)

        # Remove the item from the open set

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            node = Node(current.x + motion[i][0], current.y + motion[i][1], current.cost + motion[i][2], c_id)
            n_id = _calc_index(node, x_w, min_x, min_y)

            if n_id in closed_set:
                continue

            if not _verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                continue

            if n_id not in open_set:
                open_set[n_id] = node  # Discover a new node
                heapq.heappush(priority_queue, (node.cost, _calc_index(node, x_w, min_x, min_y)))
            else:
                if open_set[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    open_set[n_id] = node
                    heapq.heappush(priority_queue, (node.cost, _calc_index(node, x_w, min_x, min_y)))

    return closed_set


def _verify_node(node: Node, obstacle_map: list[list[bool]], min_x: int, min_y: int, max_x: int, max_y: int) -> bool:
    if node.x < min_x:
        return False
    elif node.y < min_y:
        return False
    elif node.x >= max_x:
        return False
    elif node.y >= max_y:
        return False

    if obstacle_map[node.x][node.y]:
        return False

    return True


def _calc_obstacle_map(
    obstacle_x_list: list[float],
    obstacle_y_list: list[float],
    resolution: float,
    robot_radius: float,
) -> tuple[list[list[bool]], int, int, int, int, int, int]:
    min_x = round(min(obstacle_x_list))
    min_y = round(min(obstacle_y_list))
    max_x = round(max(obstacle_x_list))
    max_y = round(max(obstacle_y_list))

    x_width = round(max_x - min_x)
    y_width = round(max_y - min_y)

    # obstacle map generation
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    for ix in range(x_width):
        x = ix + min_x
        for iy in range(y_width):
            y = iy + min_y
            #  print(x, y)
            for iox, ioy in zip(obstacle_x_list, obstacle_y_list):
                d = math.hypot(iox - x, ioy - y)
                if d <= robot_radius / resolution:
                    obstacle_map[ix][iy] = True
                    break

    return obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width


def _calc_index(node: Node, x_width: int, x_min: int, y_min: int) -> int:
    return (node.y - y_min) * x_width + (node.x - x_min)


def _get_motion_model() -> list[tuple[int, int, float]]:
    # dx, dy, cost
    motion = [
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (0, -1, 1),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    return motion
