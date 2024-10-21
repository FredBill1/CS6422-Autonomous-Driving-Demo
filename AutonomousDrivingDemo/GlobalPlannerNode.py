import multiprocessing as mp
from enum import Enum, auto
from multiprocessing.connection import Connection
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .global_planner.hybrid_a_star import Node, hybrid_a_star
from .modeling.Car import Car
from .modeling.Obstacles import Obstacles
from .utils.PipeRecvWorker import PipeRecvWorker
from .utils.set_high_priority import set_high_priority


class _ParentMsgType(Enum):
    PLAN = auto()
    CANCEL = auto()


class _WorkerMsgType(Enum):
    DISPLAY_SEGMENTS = auto()
    TRAJECTORY = auto()


def _worker_process(pipe: Connection, segment_collection_size: int) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    while True:
        match pipe.recv():
            case _ParentMsgType.CANCEL:
                continue
            case _ParentMsgType.PLAN, start, goal, obstacles:
                if pipe.poll():  # discard outdated data
                    continue

                display_segments: list[npt.NDArray[np.floating[Any]]] = []

                def callback(node: Node) -> bool:
                    display_segments.append(node.get_plot_trajectory())
                    if len(display_segments) < segment_collection_size:
                        return
                    if pipe.poll():
                        return True
                    pipe.send((_WorkerMsgType.DISPLAY_SEGMENTS, display_segments))
                    display_segments.clear()
                    return False

                trajectory = hybrid_a_star(start, goal, obstacles, callback)
                if not pipe.poll():
                    pipe.send((_WorkerMsgType.TRAJECTORY, trajectory))


class GlobalPlannerNode(QObject):
    finished = Signal()
    trajectory = Signal(np.ndarray)
    display_segments = Signal(list)

    def __init__(self, segment_collection_size: int, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._worker = mp.Process(target=_worker_process, args=(self._child_pipe, segment_collection_size), daemon=True)
        self._recv_worker = PipeRecvWorker(self._parent_pipe, parent=self)
        self._recv_worker.recv.connect(self._worker_recv)

    @Slot()
    def start(self) -> None:
        self._worker.start()
        set_high_priority(self._worker.pid)
        self._recv_worker.start(QThread.Priority.HighestPriority)

    @Slot(Car, Car, Obstacles)
    def plan(self, start_state: Car, goal_state: Car, obstacles: Obstacles) -> None:
        start = np.array([start_state.x, start_state.y, start_state.yaw])
        goal = np.array([goal_state.x, goal_state.y, goal_state.yaw])
        self._parent_pipe.send((_ParentMsgType.PLAN, start, goal, obstacles))

    @Slot()
    def cancel(self) -> None:
        self._parent_pipe.send(_ParentMsgType.CANCEL)

    @Slot(object)
    def _worker_recv(self, data) -> None:
        match data:
            case _WorkerMsgType.DISPLAY_SEGMENTS, display_segments:
                self.display_segments.emit(display_segments)
            case _WorkerMsgType.TRAJECTORY, trajectory:
                self.trajectory.emit(trajectory)
                if trajectory is not None:
                    self.finished.emit()
