from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot

from .constants import MAP_HEIGHT, MAP_NUM_RANDOM_OBSTACLES, MAP_STEP, MAP_WIDTH
from .modeling.Car import Car
from .modeling.Obstacles import Obstacles


def _generate_known_obstacle_coordnates() -> npt.NDArray[np.floating[Any]]:
    ox = [
        np.arange(0, MAP_WIDTH, MAP_STEP),
        np.full(np.ceil(MAP_HEIGHT / MAP_STEP).astype(int), MAP_WIDTH),
        np.arange(0, MAP_WIDTH + MAP_STEP, MAP_STEP),
        np.full(np.ceil(MAP_HEIGHT / MAP_STEP).astype(int) + 1, 0.0),
        np.full(np.ceil(MAP_WIDTH / 3 * 2 / MAP_STEP).astype(int), MAP_WIDTH / 3),
        np.full(np.ceil(MAP_HEIGHT / 3 * 2 / MAP_STEP).astype(int), 2 * MAP_WIDTH / 3),
    ]
    oy = [
        np.full(np.ceil(MAP_WIDTH / MAP_STEP).astype(int), 0.0),
        np.arange(0, MAP_HEIGHT, MAP_STEP),
        np.full(np.ceil((MAP_WIDTH + MAP_STEP) / MAP_STEP).astype(int), MAP_HEIGHT),
        np.arange(0, MAP_HEIGHT + MAP_STEP, MAP_STEP),
        np.arange(0, MAP_WIDTH / 3 * 2, MAP_STEP),
        MAP_HEIGHT - np.arange(0, MAP_HEIGHT / 3 * 2, MAP_STEP),
    ]
    return np.vstack((np.concatenate(ox), np.concatenate(oy))).T


def _generate_unknown_obstacle_coordnates() -> npt.NDArray[np.floating[Any]]:
    return np.random.uniform((0, 0), (MAP_WIDTH, MAP_HEIGHT), (MAP_NUM_RANDOM_OBSTACLES, 2))


class MapServerNode(QObject):
    known_obstacle_coordinates_updated = Signal(np.ndarray)
    new_obstacle_coordinates = Signal(np.ndarray)
    inited = Signal(Car)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    @Slot()
    def init(self) -> None:
        self._known_obstacle_coordinates = _generate_known_obstacle_coordnates()
        self._unknown_obstacle_coordinates = _generate_unknown_obstacle_coordnates()
        self._unknown_obstacles = Obstacles(self._unknown_obstacle_coordinates)
        self._havent_discovered = np.ones(len(self._unknown_obstacle_coordinates), dtype=bool)
        self.inited.emit(self._generate_random_initial_state())
        self.known_obstacle_coordinates_updated.emit(self._known_obstacle_coordinates)

    @property
    def known_obstacle_coordinates(self) -> npt.NDArray[np.floating[Any]]:
        return self._known_obstacle_coordinates

    @property
    def unknown_obstacle_coordinates(self) -> npt.NDArray[np.floating[Any]]:
        return self._unknown_obstacle_coordinates

    def _generate_random_initial_state(self) -> Car:
        obstacles = Obstacles(np.vstack((self._known_obstacle_coordinates, self._unknown_obstacle_coordinates)))
        state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
        while Car(*state).check_collision(obstacles):
            state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
        return Car(*state)

    def _lidar_scan(self, x: float, y: float) -> None:
        ids = np.array(self._unknown_obstacles.kd_tree.query_ball_point((x, y), Car.SCAN_RADIUS))
        if ids.size == 0:
            return
        ids: np.ndarray = ids[self._havent_discovered[ids]]
        if ids.size == 0:
            return
        self._havent_discovered[ids] = False
        new_obstacle_coordinates = self._unknown_obstacle_coordinates[ids]
        self._known_obstacle_coordinates = np.vstack((self._known_obstacle_coordinates, new_obstacle_coordinates))
        self.new_obstacle_coordinates.emit(new_obstacle_coordinates)
        self.known_obstacle_coordinates_updated.emit(self._known_obstacle_coordinates)

    @Slot(float, Car)
    def update(self, timestamp_s: float, state: Car) -> None:
        cy, sy = np.cos(state.yaw), np.sin(state.yaw)
        self._lidar_scan(state.x + cy * Car.BACK_TO_CENTER, state.y + sy * Car.BACK_TO_CENTER)
