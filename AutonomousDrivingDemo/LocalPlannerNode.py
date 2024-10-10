import multiprocessing as mp
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot

from .local_planner.ModelPredictiveControl import ModelPredictiveControl, MPCResult
from .modeling.Car import Car


def _worker_process(pipe, delta_time_s: float) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    mpc: Optional[ModelPredictiveControl] = None
    while True:
        data = pipe.recv()
        if data is None:
            mpc = None
        elif isinstance(data, np.ndarray):
            mpc = ModelPredictiveControl(data)
        elif isinstance(data, Car):
            pipe.send(None if mpc is None else mpc.update(data, delta_time_s))


class LocalPlannerNode(QObject):
    control = Signal(tuple)
    local_trajectory = Signal(np.ndarray)
    reference_points = Signal(np.ndarray)

    def __init__(self, delta_time_s: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._delta_time_s = delta_time_s
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._worker = mp.Process(target=_worker_process, args=(self._child_pipe, delta_time_s), daemon=True)
        self._worker.start()

    @Slot(float, Car)
    def update(self, timestamp_s: float, state: Car) -> None:
        self._parent_pipe.send(state)
        result: Optional[MPCResult] = self._parent_pipe.recv()
        if result is None:
            return
        acceleration, steer = result.controls[1]
        control = (state.velocity + acceleration * self._delta_time_s, steer)
        self.control.emit(control)
        self.local_trajectory.emit(result.states[:, :2])
        self.reference_points.emit(result.ref_states[:, :2])

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        self._parent_pipe.send(trajectory)

    @Slot()
    def cancel(self) -> None:
        self._parent_pipe.send(None)
