import multiprocessing as mp
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


def _worker_process(pipe: Connection, segment_collection_size: int) -> None:
    # use multiprocessing to bypass the GIL to prevent GUI freezes
    while True:
        data: Optional[tuple[Car, Car, Obstacles]] = pipe.recv()
        if data is None:
            continue
        display_segments: list[npt.NDArray[np.floating[Any]]] = []
        canceled = False

        def callback(node: Node) -> bool:
            display_segments.append(node.get_plot_trajectory())
            if len(display_segments) < segment_collection_size:
                return
            if pipe.poll():
                nonlocal canceled
                canceled = True
                return True
            pipe.send(display_segments)
            display_segments.clear()
            return False

        trajectory = hybrid_a_star(*data, callback)
        if not canceled:
            pipe.send(trajectory)


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
        self._parent_pipe.send((start, goal, obstacles))

    @Slot()
    def cancel(self) -> None:
        self._parent_pipe.send(None)

    @Slot(object)
    def _worker_recv(self, data: list[npt.NDArray[np.floating[Any]]] | Optional[npt.NDArray[np.floating[Any]]]) -> None:
        if isinstance(data, list):
            self.display_segments.emit(data)
        else:
            self.trajectory.emit(data)
            if data is not None:
                self.finished.emit()
