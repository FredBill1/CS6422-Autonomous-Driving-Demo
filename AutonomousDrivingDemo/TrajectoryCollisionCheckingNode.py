from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot
from scipy.spatial import KDTree

from .modeling.Car import Car
from .modeling.Obstacles import Obstacles

DISCARD_FIRST_N = 5


class TrajectoryCollisionChecker:
    def __init__(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        assert trajectory.ndim == 2 and trajectory.shape[1] == 3, "trajectory must be 2D array having [[x, y, yaw]]"
        self._trajectory = trajectory

        # Calculate the trajectory of the center of the car, instead of the center of the rear axle
        xy, yaw = trajectory[:, :2], trajectory[:, 2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        xy = (xy.T + [Car.BACK_TO_CENTER * cy, Car.BACK_TO_CENTER * sy]).T
        self._trajectory_kd_tree = KDTree(xy)

    def check(self, obstacles: Obstacles) -> bool:
        indices = self._trajectory_kd_tree.query_ball_tree(obstacles.kd_tree, Car.COLLISION_RADIUS)
        for i, ids in enumerate(indices):
            if not ids:
                continue
            if Car(*self._trajectory[i]).check_collision(obstacles.coordinates[ids]):
                return True
        return False


class TrajectoryCollisionCheckingNode(QObject):
    collided = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._checker: Optional[TrajectoryCollisionChecker] = None
        self._known_obstacles: Optional[npt.NDArray[np.floating[Any]]] = None

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        if trajectory is None:
            self._checker = None
            return
        self._checker = TrajectoryCollisionChecker(trajectory[DISCARD_FIRST_N:, :3])
        if self._known_obstacles is None:
            return
        self.check_collision(self._known_obstacles)

    @Slot(np.ndarray)
    def set_known_obstacles(self, known_obstacles) -> None:
        self._known_obstacles = known_obstacles

    @Slot(np.ndarray)
    def check_collision(self, obstacle_coordinates: npt.NDArray[np.floating[Any]]) -> None:
        if self._checker is None:
            return
        if self._checker.check(Obstacles(obstacle_coordinates)):
            self.collided.emit()

    @Slot()
    def cancel(self) -> None:
        self._checker = None
