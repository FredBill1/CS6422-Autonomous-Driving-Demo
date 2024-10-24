from enum import Enum, auto
from multiprocessing.connection import Connection
from typing import Any, Optional, override

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Qt, QThread, QTimerEvent, Signal, Slot

from .local_planner.ModelPredictiveControl import ModelPredictiveControl, MPCResult
from .modeling.Car import Car
from .utils.ProcessWithPipe import ProcessWithPipe


class _ParentMsgType(Enum):
    TRAJECTORY = auto()
    STATE = auto()
    BRAKE = auto()
    CANCEL = auto()


def _worker_process(pipe: Connection, delta_time_s: float) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    mpc: Optional[ModelPredictiveControl] = None
    while True:
        match pipe.recv():
            case _ParentMsgType.CANCEL:
                mpc = None
            case _ParentMsgType.TRAJECTORY, trajectory:
                mpc = ModelPredictiveControl(trajectory)
            case _ParentMsgType.BRAKE:
                if mpc is not None:
                    mpc.brake()
            case _ParentMsgType.STATE, (timestamp_s, state):
                if pipe.poll() or mpc is None:  # discard outdated data
                    continue
                pipe.send((timestamp_s, state, mpc.update(state, delta_time_s)))


class LocalPlannerNode(QObject):
    control_sequence = Signal(tuple)
    local_trajectory = Signal(np.ndarray)
    reference_points = Signal(np.ndarray)
    brake_trajectory = Signal(np.ndarray)

    def __init__(self, delta_time_s: float, update_interval_s: float, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state: Optional[tuple[float, Car]] = None
        self._delta_time_s = delta_time_s
        self._worker = ProcessWithPipe(_worker_process, args=(delta_time_s,), parent=self)
        self._worker.recv.connect(self._worker_recv)
        self._update_interval = int(update_interval_s * 1000)

    @Slot()
    def start(self) -> None:
        self._worker.start(QThread.Priority.HighestPriority)
        self.startTimer(self._update_interval, Qt.TimerType.PreciseTimer)

    @Slot(float, Car)
    def set_state(self, timestamp_s: float, state: Car) -> None:
        self._state = (timestamp_s, state)

    @Slot(np.ndarray)
    def set_trajectory(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        if trajectory is not None:
            self._worker.send((_ParentMsgType.TRAJECTORY, trajectory))
        else:
            self._worker.send(_ParentMsgType.BRAKE)

    @Slot()
    def brake(self) -> None:
        self._worker.send(_ParentMsgType.BRAKE)

    @Slot()
    def cancel(self) -> None:
        self._worker.send(_ParentMsgType.CANCEL)

    @override
    def timerEvent(self, _: QTimerEvent) -> None:
        if self._state is not None:
            self._worker.send((_ParentMsgType.STATE, self._state))

    @Slot(tuple)
    def _worker_recv(self, data: tuple[float, Car, MPCResult]) -> None:
        timestamp_s, state, result = data
        timestamps = np.arange(len(result.controls)) * self._delta_time_s + timestamp_s
        velocities = state.velocity + np.cumsum(result.controls[:, 0] * self._delta_time_s)
        steers = result.controls[:, 1]
        control_sequence = np.column_stack((timestamps, velocities, steers))
        self.control_sequence.emit(control_sequence)
        self.local_trajectory.emit(result.states[:, :2])
        self.reference_points.emit(result.ref_states)
        self.brake_trajectory.emit(result.brake_trajectory)
