import heapq
from collections.abc import Callable, Generator
from itertools import islice, product
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from rsplan import Path as RSPath
from rsplan.planner import _solve_path as solve_rspath

from ..modeling.Car import Car
from ..modeling.Obstacles import ObstacleGrid, Obstacles
from ..utils.SupportsBool import SupportsBool
from ..utils.wrap_angle import wrap_angle

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.5  # [m] path interpolate resolution
MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5  # [m] path interpolate distance
NUM_STEER_COMMANDS = 10  # number of steer command

REEDS_SHEPP_MAX_DISTANCE = 10.0  # maximum distance to use Reeds-Shepp path

SWITCH_DIRECTION_COST = 25.0  # switch direction cost
BACKWARDS_COST = 4.0  # backward movement cost
STEER_CHANGE_COST = 3.0  # steer angle change cost
STEER_COST = 1.5  # steer angle cost per distance
H_DIST_COST = 2.0  # Heuristic distance cost
H_YAW_COST = 3.0 / np.deg2rad(45)  # Heuristic yaw difference cost
H_COLLISION_COST = 1e4  # collision cost when calculating heuristic

# if True, return the Reeds-Shepp path immediately when it is found
# otherwise, continue the A* search to find a better path (may be much slower)
RETURN_RS_PATH_IMMEDIATELY = True

STEER_COMMANDS = np.unique(
    np.concatenate([np.linspace(-Car.TARGET_MAX_STEER, Car.TARGET_MAX_STEER, NUM_STEER_COMMANDS), [0.0]])
)


MOVEMENTS = tuple((di, dj, np.sqrt(di**2 + dj**2)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)


def _distance_heuristic(grid: ObstacleGrid, goal_xy: npt.ArrayLike) -> ObstacleGrid:
    "Dijkstra's algorithm to calculate the distance from each grid cell to the goal"
    H, W = grid.grid.shape
    dist = np.full((H, W), H_COLLISION_COST)
    ij = grid.calc_index(goal_xy)
    dist[ij] = 0
    pq = [(0, ij)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, cost in MOVEMENTS:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not grid.grid[ni, nj] and d + cost < dist[ni, nj]:
                dist[ni, nj] = d + cost
                heapq.heappush(pq, (d + cost, (ni, nj)))
    return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)


class SimplePath(NamedTuple):
    ijk: tuple[int, int, int]  # grid index
    trajectory: npt.NDArray[np.floating[Any]]  # [[x(m), y(m), yaw(rad)]]
    direction: Literal[1, 0, -1]  # direction, 1 forward, -1 backward, 0 initial
    steer: float  # [rad], [-TARGET_MAX_STEER, TARGET_MAX_STEER]


class Node(NamedTuple):
    path: SimplePath | RSPath
    cost: float
    h_cost: float
    parent: Optional["Node"]

    def __lt__(self, other: "Node") -> bool:
        return (self.h_cost + self.cost, self.cost) < (other.h_cost + other.cost, other.cost)

    def get_plot_trajectory(self) -> npt.NDArray[np.floating[Any]]:
        "Get the trajectory coordinates for visualization"
        trajectory = (
            np.array([[p.x, p.y] for p in self.path.waypoints()])
            if isinstance(self.path, RSPath)
            else self.path.trajectory[:, :2]
        )
        if self.parent is not None:
            trajectory = np.vstack((self.parent.path.trajectory[-1, :2], trajectory))
        return trajectory


def hybrid_a_star(
    start: npt.NDArray[np.floating[Any]],
    goal: npt.NDArray[np.floating[Any]],
    obstacles: Obstacles,
    cancel_callback: Optional[Callable[[Node], SupportsBool]] = None,
) -> Optional[npt.NDArray[np.floating[Any]]]:
    assert start.shape == (3,) or (
        len(start.shape) == 2 and start.shape[1] == 4
    ), "Start must be a 1D array of shape (3) representing [x, y, yaw] or a 2D array of shape (N, 4) representing [x, y, yaw, velocity]"
    assert goal.shape == (3,), "Goal must be a 1D array of shape (3) representing [x, y, yaw]"

    if Car(*goal).check_collision(obstacles):
        return None

    start_is_point = start.shape == (3,)
    start_collided = Car(*start).check_collision(obstacles) if start_is_point else False

    # Downsample the obstacle map to a grid
    obstacle_grid = obstacles.downsampling_to_grid(
        XY_GRID_RESOLUTION, min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH) / 2
    )

    # Precompute the distance to the goal from each grid cell, where the distance will be used as a heuristic
    heuristic_grid = _distance_heuristic(obstacle_grid, goal[:2])
    N, M = heuristic_grid.grid.shape
    K = int(2 * np.pi / YAW_GRID_RESOLUTION)

    # Used to record the path and cost for each grid cell at A* search stage,
    # where dp[y][x][yaw] is the Node object for the grid cell (x, y) with yaw angle yaw
    dp = np.full((N, M, K), None, dtype=Node)

    def calc_ijk(x: float, y: float, yaw: float) -> tuple[int, int, int]:
        "[x, y, yaw] -> [i, j, k] for dp"
        i, j = heuristic_grid.calc_index([x, y])
        k = int(wrap_angle(yaw, zero_to_2pi=True) // YAW_GRID_RESOLUTION)
        return i, j, k

    def generate_neighbour(cur: Node, direction: int, steer: float) -> Optional[Node]:
        "Generate a neighbour node of the current node, given the direction and steer angle"

        # Simulate the car movement for MOTION_DISTANCE, with a interval of MOTION_RESOLUTION,
        # check if the car will collide with the obstacles during the movement
        car = Car(*cur.path.trajectory[-1, :3], velocity=float(direction), steer=steer)
        trajectory = []
        for _ in range(int(MOTION_DISTANCE / MOTION_RESOLUTION)):
            car.update(MOTION_RESOLUTION)
            if not start_collided and car.check_collision(obstacles):
                return None
            trajectory.append([car.x, car.y, car.yaw])

        i, j, k = calc_ijk(car.x, car.y, car.yaw)
        if not (0 <= i < N and 0 <= j < M):
            print(f"Out of grid, please add more obstacles to fill the boundary: {i=} {j=}")
            return None

        # Calculate the cost from the start to this neighbour node
        distance_cost = MOTION_DISTANCE if direction == 1 else MOTION_DISTANCE * BACKWARDS_COST
        switch_direction_cost = (
            SWITCH_DIRECTION_COST if cur.path.direction != 0 and direction != cur.path.direction else 0.0
        )
        steer_change_cost = STEER_CHANGE_COST * np.abs(steer - cur.path.steer)
        steer_cost = STEER_COST * np.abs(steer) * MOTION_DISTANCE
        cost = cur.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        # Calculate the heuristic cost from this neighbour node to the goal
        h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]
        h_yaw_cost = H_YAW_COST * np.abs(wrap_angle(goal[2] - car.yaw))
        h_cost = h_dist_cost + h_yaw_cost

        return Node(SimplePath((i, j, k), np.array(trajectory), direction, steer), cost, h_cost, cur)

    def generate_neighbours(cur: Node) -> Generator[Node, None, None]:
        "Generate all possible neighbours of the current node"
        nonlocal start_collided
        for direction, steer in product([1, -1], STEER_COMMANDS):
            if (res := generate_neighbour(cur, direction, steer)) is not None:
                yield res
        start_collided = False

    def generate_rspath(node: Node) -> Optional[Node]:
        """
        Try to generate a Path from the current node directly to the goal using Reeds-Shepp curves,
        which will speed up the search process when the node is close to the goal and heuristics
        are not enough to guide the search.
        """

        def check(path: RSPath) -> bool:
            for x, y, yaw in zip(*path.coordinates_tuple()):
                if Car(x, y, yaw).check_collision(obstacles):
                    return False
            return True

        def calc_rspath_cost(path: RSPath) -> float:
            """
            Same logic to calculate the cost of a path in the `generate_neighbour` above, except that:

            1. the heuristic cost is 0, since the Reeds-Shepp path is directly from the current node to the goal.
            2. the cost of the path is calculated as the sum of the cost of each segment in the path.
            """
            last_direction = node.path.direction
            last_steer = node.path.steer

            distance_cost = 0.0
            switch_direction_cost = 0.0
            steer_change_cost = 0.0
            steer_cost = 0.0
            for segment in path.segments:
                length = abs(segment.length)
                distance_cost += length if segment.direction == 1 else length * BACKWARDS_COST
                if last_direction != 0 and segment.direction != last_direction:
                    switch_direction_cost += SWITCH_DIRECTION_COST
                last_direction = segment.direction
                steer = {"left": Car.TARGET_MAX_STEER, "right": -Car.TARGET_MAX_STEER, "straight": 0.0}[segment.type]
                steer_change_cost += STEER_CHANGE_COST * np.abs(steer - last_steer)
                last_steer = steer
                steer_cost += STEER_COST * np.abs(steer) * length
            return distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        # generate all possible Reeds-Shepp pathes
        pathes = solve_rspath(
            tuple(node.path.trajectory[-1, :3]), tuple(goal), Car.TARGET_MIN_TURNING_RADIUS, MOTION_RESOLUTION
        )

        # filter out the pathes that collide with the obstacles
        pathes = filter(check, pathes)

        # calculate the cost of each path
        pathes = ((path, calc_rspath_cost(path)) for path in pathes)

        # return the path with the minimum cost
        if (ret := min(pathes, key=lambda t: t[1], default=None)) is None:
            return None
        path, cost = ret
        return Node(path, node.cost + cost, 0.0, node)

    def traceback_path(node: Node) -> npt.NDArray[np.floating[Any]]:
        """
        Traceback the path from the goal to the start, to get the final trajectory

        returns [[x(m), y(m), yaw(rad), direction(1, -1)]]
        """
        segments = []
        while node is not None:
            path = node.path
            if isinstance(path, SimplePath):
                if path.trajectory.shape[1] == 4:
                    segments.append(path.trajectory)
                else:
                    segments.append(np.hstack((path.trajectory, np.full_like(path.trajectory[:, :1], path.direction))))
            else:
                # RSPath contains the start point, so we skip it using islice
                segments.append([[p.x, p.y, p.yaw, p.driving_direction] for p in islice(path.waypoints(), 1, None)])
            node = node.parent
        segments.reverse()
        trajectory = np.vstack(segments)
        trajectory[0, 3] = trajectory[1, 3] if trajectory.shape[0] > 1 else 1  # set the initial driving direction
        return trajectory

    if start_is_point:
        start_ijk = calc_ijk(*start)
        start_path = SimplePath(start_ijk, np.array([start]), 0, 0.0)
    else:
        xy = start[:, :2]
        mask = (xy[:-1] != xy[1:]).any(axis=1)  # remove consecutive identical points
        start = start[np.concatenate(([True], mask))]
        start_ijk = calc_ijk(*start[-1, :3])
        start[:, 3] = np.sign(start[0, 3])
        steer = 0.0
        if start.shape[0] >= 2 and (l := np.linalg.norm(start[-1, :2] - start[-2, :2])):
            steer = np.arctan(Car.WHEEL_BASE * (start[-1, 2] - start[-2, 2]) / l)
        start_path = SimplePath(start_ijk, start, start[0, 3], steer)
    start_node = Node(start_path, 0.0, H_DIST_COST * heuristic_grid.grid[start_ijk[:2]], None)

    dp[start_ijk] = start_node
    pq = [start_node]
    while pq:  # A* search (Similar to Dijkstra's algorithm, but with a heuristic cost added)
        cur = heapq.heappop(pq)
        if isinstance(cur.path, RSPath):
            if cancel_callback is not None and cancel_callback(cur):
                return None  # canceled
            return traceback_path(cur)

        if cur.cost > dp[cur.path.ijk].cost:
            continue

        if cancel_callback is not None and cancel_callback(cur):
            return None  # canceled

        if np.linalg.norm(cur.path.trajectory[-1, :2] - goal[:2]) <= REEDS_SHEPP_MAX_DISTANCE:
            if (rsnode := generate_rspath(cur)) is not None:
                if RETURN_RS_PATH_IMMEDIATELY:
                    return traceback_path(rsnode)
                heapq.heappush(pq, rsnode)

        for neighbour in generate_neighbours(cur):
            if dp[neighbour.path.ijk] is None or neighbour.cost < dp[neighbour.path.ijk].cost:
                dp[neighbour.path.ijk] = neighbour
                heapq.heappush(pq, neighbour)
    return None
