from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot

from .local_planner.ModelPredictiveControl import ModelPredictiveControl
from .modeling.Car import Car


class LocalPlannerNode(QObject):
    control = Signal(tuple)
    local_trajectory = Signal(np.ndarray)
    reference_points = Signal(np.ndarray)

    def __init__(self, delta_time_s: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._mpc: Optional[ModelPredictiveControl] = None
        self._delta_time_s = delta_time_s

    @Slot(Car)
    def update(self, timestamp_s: float, state: Car) -> None:
        if self._mpc is None:
            return
        result = self._mpc.update(state, self._delta_time_s)
        acceleration, steer = result.controls[1]
        control = (state.velocity + acceleration * self._delta_time_s, steer)
        self.control.emit(control)
        self.local_trajectory.emit(result.states[:, :2])
        self.reference_points.emit(result.ref_states[:, :2])

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        self._mpc = ModelPredictiveControl(trajectory) if trajectory is not None else None

    @Slot()
    def cancel(self) -> None:
        self._mpc = None
