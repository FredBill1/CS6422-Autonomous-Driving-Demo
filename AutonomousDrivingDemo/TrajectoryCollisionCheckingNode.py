from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot
from scipy.spatial import KDTree

from .modeling.Car import Car
from .modeling.Obstacles import Obstacles


class TrajectoryCollisionChecker:
    def __init__(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        assert trajectory.ndim == 2 and trajectory.shape[1] == 3, "trajectory must be 2D array having [[x, y, yaw]]"
        self._trajectory = trajectory
        self._trajectory_kd_tree = KDTree(trajectory[:, :2])

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

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        self._checker = TrajectoryCollisionChecker(trajectory[:, :3]) if trajectory is not None else None

    @Slot(np.ndarray)
    def check_collision(self, obstacle_coordinates: npt.NDArray[np.floating[Any]]) -> None:
        if self._checker is None:
            return
        if self._checker.check(Obstacles(obstacle_coordinates)):
            self.collided.emit()
            self._checker = None
