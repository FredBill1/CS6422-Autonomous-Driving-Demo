import multiprocessing as mp
from enum import Enum, auto
from multiprocessing.connection import Connection
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from .local_planner.ModelPredictiveControl import ModelPredictiveControl, MPCResult
from .modeling.Car import Car
from .utils.PipeRecvWorker import PipeRecvWorker
from .utils.set_high_priority import set_high_priority


class _ParentMsgType(Enum):
    TRAJECTORY = auto()
    STATE = auto()
    CANCEL = auto()


def _worker_process(pipe: Connection, delta_time_s: float) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    mpc: Optional[ModelPredictiveControl] = None
    while True:
        match pipe.recv():
            case _ParentMsgType.CANCEL, None:
                mpc = None
            case _ParentMsgType.TRAJECTORY, trajectory:
                mpc = ModelPredictiveControl(trajectory)
            case _ParentMsgType.STATE, (timestamp_s, state):
                if pipe.poll() or mpc is None:  # discard outdated data
                    continue
                pipe.send((timestamp_s, state, mpc.update(state, delta_time_s)))


class LocalPlannerNode(QObject):
    control_sequence = Signal(tuple)
    local_trajectory = Signal(np.ndarray)
    reference_points = Signal(np.ndarray)

    def __init__(self, delta_time_s: float, update_interval_s: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: Optional[tuple[float, Car]] = None
        self._delta_time_s = delta_time_s
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._worker = mp.Process(target=_worker_process, args=(self._child_pipe, delta_time_s), daemon=True)
        self._recv_worker = PipeRecvWorker(self._parent_pipe, parent=self)
        self._recv_worker.recv.connect(self._worker_recv)

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update)
        self._update_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._update_timer.setInterval(int(update_interval_s * 1000))

        self._brake = False

    @Slot()
    def start(self) -> None:
        self._worker.start()
        set_high_priority(self._worker.pid)
        self._recv_worker.start(QThread.Priority.HighestPriority)
        self._update_timer.start()

    @Slot(float, Car)
    def set_state(self, timestamp_s: float, state: Car) -> None:
        self._state = (timestamp_s, state)

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        if trajectory is not None:
            self._parent_pipe.send((_ParentMsgType.TRAJECTORY, trajectory))
            self._brake = False
        else:
            self._brake = True

    @Slot()
    def brake(self) -> None:
        self._brake = True

    @Slot()
    def cancel(self) -> None:
        self._parent_pipe.send((_ParentMsgType.CANCEL, None))

    @Slot()
    def _update(self) -> None:
        if self._state is not None:
            self._parent_pipe.send((_ParentMsgType.STATE, self._state))

    @Slot(tuple)
    def _worker_recv(self, data: tuple[float, Car, MPCResult]) -> None:
        timestamp_s, state, result = data
        timestamps = np.arange(len(result.controls)) * self._delta_time_s + timestamp_s
        if not self._brake:
            velocities = state.velocity + np.cumsum(result.controls[:, 0] * self._delta_time_s)
        else:
            velocities = np.zeros_like(result.controls[:, 0])
        steers = result.controls[:, 1]
        control_sequence = np.column_stack((timestamps, velocities, steers))
        self.control_sequence.emit(control_sequence)
        self.local_trajectory.emit(result.states[:, :3])
        self.reference_points.emit(result.ref_states[:, :2])
