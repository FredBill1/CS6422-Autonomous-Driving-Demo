import heapq
from collections.abc import Callable, Generator
from itertools import product
from typing import Any, Literal, NamedTuple, Optional, Protocol

import numpy as np
import numpy.typing as npt
from rsplan import Path as RSPath
from rsplan.planner import _solve_path as solve_rspath

from ..modeling.Car import Car
from ..modeling.Obstacles import ObstacleGrid, Obstacles
from ..utils.wrap_angle import wrap_angle

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.5  # [m] path interpolate resolution
MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5  # [m] path interpolate distance
NUM_STEER_COMMANDS = 20  # number of steer command

SWITCH_DIRECTION_COST = 25.0  # switch direction cost
BACKWARDS_COST = 4.0  # backward penalty cost
STEER_CHANGE_COST = 3.0  # steer angle change cost
STEER_COST = 3.0  # steer angle cost
H_COST = 2.0  # Heuristic cost
H_COLLISION_COST = 1e4  # collision cost when calculating heuristic

STEER_COMMANDS = np.unique(
    np.concatenate([np.linspace(-Car.TARGET_MAX_STEER, Car.TARGET_MAX_STEER, NUM_STEER_COMMANDS), [0.0]])
)


MOVEMENTS = tuple((di, dj, np.sqrt(di**2 + dj**2)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)


def _distance_heuristic(grid: ObstacleGrid, goal_xy: npt.ArrayLike) -> ObstacleGrid:
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
        trajectory = (
            np.array([[p.x, p.y] for p in self.path.waypoints()])
            if isinstance(self.path, RSPath)
            else self.path.trajectory[:, :2]
        )
        if self.parent is not None:
            trajectory = np.vstack((self.parent.path.trajectory[-1, :2], trajectory))
        return trajectory


class SupportsBool(Protocol):
    def __bool__(self) -> bool: ...


def hybrid_a_star(
    start: npt.NDArray[np.floating[Any]],
    goal: npt.NDArray[np.floating[Any]],
    obstacles: Obstacles,
    cancel_callback: Optional[Callable[[Node], SupportsBool]] = None,
) -> Optional[npt.NDArray[np.floating[Any]]]:
    assert start.shape == (3,) and goal.shape == (3,), "Start and goal must be a 1D array of shape (3)"

    if Car(*goal).check_collision(obstacles):
        return None

    obstacle_grid = obstacles.downsampling_to_grid(
        XY_GRID_RESOLUTION, min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH) / 2
    )
    heuristic_grid = _distance_heuristic(obstacle_grid, goal[:2])
    N, M = heuristic_grid.grid.shape
    K = int(2 * np.pi / YAW_GRID_RESOLUTION)
    dp = np.full((N, M, K), None, dtype=object)

    def calc_ijk(x: float, y: float, yaw: float) -> tuple[int, int, int]:
        i, j = heuristic_grid.calc_index([x, y])
        k = int(wrap_angle(yaw, zero_to_2pi=True) // YAW_GRID_RESOLUTION)
        return i, j, k

    def generate_neighbour(cur: Node, direction: int, steer: float) -> Optional[Node]:
        car = Car(*cur.path.trajectory[-1], velocity=float(direction), steer=steer)
        trajectory = []
        for _ in range(int(MOTION_DISTANCE / MOTION_RESOLUTION)):
            car.update(MOTION_RESOLUTION)
            if car.check_collision(obstacles):
                return None
            trajectory.append([car.x, car.y, car.yaw])

        i, j, k = calc_ijk(car.x, car.y, car.yaw)
        if not (0 <= i < N and 0 <= j < M):
            print(f"Out of grid, please add more obstacles to fill the boundary: {i=} {j=}")
            return None

        distance_cost = MOTION_DISTANCE if direction == 1 else MOTION_DISTANCE * BACKWARDS_COST
        switch_direction_cost = (
            SWITCH_DIRECTION_COST if cur.path.direction != 0 and direction != cur.path.direction else 0.0
        )
        steer_change_cost = STEER_CHANGE_COST * np.abs(steer - cur.path.steer)
        steer_cost = STEER_COST * np.abs(steer)
        cost = cur.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost
        h_cost = H_COST * heuristic_grid.grid[i, j]

        return Node(SimplePath((i, j, k), np.array(trajectory), direction, steer), cost, h_cost, cur)

    def generate_neighbours(cur: Node) -> Generator[Node, None, None]:
        for direction, steer in product([1, -1], STEER_COMMANDS):
            if (res := generate_neighbour(cur, direction, steer)) is not None:
                yield res

    def generate_rspath(node: Node) -> Optional[Node]:
        def check(path: RSPath) -> bool:
            for x, y, yaw in zip(*path.coordinates_tuple()):
                if Car(x, y, yaw).check_collision(obstacles):
                    return False
            return True

        def calc_rspath_cost(path: RSPath) -> float:
            last_direction = node.path.direction
            last_steer = node.path.trajectory[-1, 2]

            distance_cost = 0.0
            switch_direction_cost = 0.0
            steer_change_cost = 0.0
            steer_cost = 0.0
            for segment in path.segments:
                distance_cost += segment.length if segment.direction == 1 else segment.length * BACKWARDS_COST
                if last_direction != 0 and segment.direction != last_direction:
                    switch_direction_cost += SWITCH_DIRECTION_COST
                    last_direction = segment.direction
                steer = {"left": Car.TARGET_MAX_STEER, "right": -Car.TARGET_MAX_STEER, "straight": 0.0}[segment.type]
                steer_change_cost += STEER_CHANGE_COST * np.abs(steer - last_steer)
                last_steer = steer
                steer_cost += STEER_COST * np.abs(steer)
            return distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        pathes = solve_rspath(
            tuple(node.path.trajectory[-1]), tuple(goal), Car.TARGET_MIN_TURNING_RADIUS, MOTION_RESOLUTION
        )
        pathes = filter(check, pathes)
        pathes = ((path, calc_rspath_cost(path)) for path in pathes)
        if (ret := min(pathes, key=lambda t: t[1], default=None)) is None:
            return None
        path, cost = ret
        return Node(path, node.cost + cost, 0.0, node)

    def traceback_path(node: Node, rspath: RSPath) -> npt.NDArray[np.floating[Any]]:
        # returns [[x(m), y(m), yaw(rad), direction(1, -1)]]
        segments = []
        while node is not None:
            path: SimplePath = node.path
            segments.append(np.hstack((path.trajectory, np.full_like(path.trajectory[:, :1], path.direction))))
            node = node.parent
        segments.reverse()
        segments.append([[p.x, p.y, p.yaw, p.driving_direction] for p in rspath.waypoints()])
        trajectory = np.vstack(segments)
        if trajectory.shape[0] > 1:
            trajectory[0, 3] = trajectory[1, 3]  # set the initial driving direction
        return trajectory

    start_ijk = calc_ijk(*start)
    start_node = Node(
        SimplePath(start_ijk, np.array([start]), 0, 0.0), 0.0, H_COST * heuristic_grid.grid[start_ijk[:2]], None
    )
    dp[start_ijk] = start_node
    pq = [start_node]
    while pq:
        cur = heapq.heappop(pq)
        if isinstance(cur.path, RSPath):
            if cancel_callback is not None and cancel_callback(cur):
                return None  # canceled
            return traceback_path(cur.parent, cur.path)

        if cur.cost > dp[cur.path.ijk].cost:
            continue

        if cancel_callback is not None and cancel_callback(cur):
            return None  # canceled

        if (rsnode := generate_rspath(cur)) is not None:
            heapq.heappush(pq, rsnode)

        for neighbour in generate_neighbours(cur):
            if dp[neighbour.path.ijk] is None or neighbour.cost < dp[neighbour.path.ijk].cost:
                dp[neighbour.path.ijk] = neighbour
                heapq.heappush(pq, neighbour)
    return None
