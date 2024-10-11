import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from .local_planner.ModelPredictiveControl import ModelPredictiveControl, MPCResult
from .modeling.Car import Car
from .utils.PipeRecvWorker import PipeRecvWorker


def _worker_process(pipe: Connection, delta_time_s: float) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    mpc: Optional[ModelPredictiveControl] = None
    while True:
        data = pipe.recv()
        if data is None:
            mpc = None
        elif isinstance(data, np.ndarray):
            mpc = ModelPredictiveControl(data)
        elif isinstance(data, Car):
            pipe.send(None if mpc is None else (data, mpc.update(data, delta_time_s)))


class LocalPlannerNode(QObject):
    control = Signal(tuple)
    local_trajectory = Signal(np.ndarray)
    reference_points = Signal(np.ndarray)

    def __init__(self, delta_time_s: float, update_interval_s: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: Optional[Car] = None
        self._delta_time_s = delta_time_s
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._worker = mp.Process(target=_worker_process, args=(self._child_pipe, delta_time_s), daemon=True)
        self._recv_worker = PipeRecvWorker(self._parent_pipe, parent=self)
        self._recv_worker.recv.connect(self._worker_recv)

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update)
        self._update_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._update_timer.setInterval(int(update_interval_s * 1000))

    @Slot()
    def start(self) -> None:
        self._worker.start()
        self._recv_worker.start()
        self._update_timer.start()

    @Slot(float, Car)
    def set_state(self, timestamp_s: float, state: Car) -> None:
        self._state = state

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        self._parent_pipe.send(trajectory)

    @Slot()
    def cancel(self) -> None:
        self._parent_pipe.send(None)

    @Slot()
    def _update(self) -> None:
        self._parent_pipe.send(self._state)

    @Slot(tuple)
    def _worker_recv(self, data: Optional[tuple[Car, MPCResult]]) -> None:
        if data is None:
            return
        state, result = data
        acceleration, steer = result.controls[1]
        control = (state.velocity + acceleration * self._delta_time_s, steer)
        self.control.emit(control)
        self.local_trajectory.emit(result.states[:, :2])
        self.reference_points.emit(result.ref_states[:, :2])
