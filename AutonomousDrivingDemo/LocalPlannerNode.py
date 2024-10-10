import multiprocessing as mp
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot

from .local_planner.ModelPredictiveControl import ModelPredictiveControl, MPCResult
from .modeling.Car import Car


def _worker_process(input_pipe: mp.SimpleQueue, output_pipe: mp.SimpleQueue, delta_time_s: float) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    mpc: Optional[ModelPredictiveControl] = None
    while True:
        data = input_pipe.get()
        if data is None:
            mpc = None
        elif isinstance(data, np.ndarray):
            mpc = ModelPredictiveControl(data)
        elif isinstance(data, Car):
            output_pipe.put(None if mpc is None else mpc.update(data, delta_time_s))


class LocalPlannerNode(QObject):
    control = Signal(tuple)
    local_trajectory = Signal(np.ndarray)
    reference_points = Signal(np.ndarray)

    def __init__(self, delta_time_s: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._delta_time_s = delta_time_s
        self._input_pipe = mp.SimpleQueue()
        self._output_pipe = mp.SimpleQueue()
        self._worker = mp.Process(
            target=_worker_process, args=(self._input_pipe, self._output_pipe, delta_time_s), daemon=True
        )
        self._worker.start()

    @Slot(float, Car)
    def update(self, timestamp_s: float, state: Car) -> None:
        self._input_pipe.put(state)
        result: Optional[MPCResult] = self._output_pipe.get()
        if result is None:
            return
        acceleration, steer = result.controls[1]
        control = (state.velocity + acceleration * self._delta_time_s, steer)
        self.control.emit(control)
        self.local_trajectory.emit(result.states[:, :2])
        self.reference_points.emit(result.ref_states[:, :2])

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        self._input_pipe.put(trajectory)

    @Slot()
    def cancel(self) -> None:
        self._input_pipe.put(None)
