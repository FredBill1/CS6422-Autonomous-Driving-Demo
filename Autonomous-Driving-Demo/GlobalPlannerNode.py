import multiprocessing as mp
from typing import Any, Optional

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .global_planner.hybrid_a_star import Node, hybrid_a_star
from .modeling.Car import Car
from .modeling.Obstacles import Obstacles


def _worker_process(pipe: mp.SimpleQueue, cancel_event, segment_collection_size: int, *args) -> None:
    display_segments: list[npt.NDArray[np.floating[Any]]] = []

    def callback(node: Node) -> bool:
        display_segments.append(node.get_plot_trajectory())
        if len(display_segments) < segment_collection_size:
            return
        if cancel_event.is_set():
            return True
        pipe.put(LineCollection(display_segments, colors=mcolors.TABLEAU_COLORS))
        display_segments.clear()
        return False

    path = hybrid_a_star(*args, callback)
    pipe.put(path)


class _GlobalPlannerNodeWorker(QThread):
    result = Signal(np.ndarray, bool)
    display_segments = Signal(LineCollection)

    def __init__(self, *args, segment_collection_size: int, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._args = args
        self._segment_collection_size = segment_collection_size

    def run(self) -> None:
        # use multiprocessing to bypass the GIL to prevent GUI freezes
        pipe = mp.SimpleQueue()
        cancel_event = mp.Event()
        p = mp.Process(target=_worker_process, args=(pipe, cancel_event, self._segment_collection_size, *self._args))
        p.start()
        while True:
            data = pipe.get()  # blocking
            if not isinstance(data, LineCollection):
                break
            if self.isInterruptionRequested():
                cancel_event.set()
            else:
                self.display_segments.emit(data)
        p.join()
        pipe.close()
        self.result.emit(data, self.isInterruptionRequested())


class GlobalPlannerNode(QObject):
    trajectory = Signal(np.ndarray)
    display_segments = Signal(LineCollection)

    def __init__(self, segment_collection_size: int, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[_GlobalPlannerNodeWorker] = None
        self._segment_collection_size = segment_collection_size

    @Slot(Car, Car, Obstacles)
    def plan(self, start_state: Car, goal_state: Car, obstacles: Obstacles) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
            self._worker.wait()
        start = np.array([start_state.x, start_state.y, start_state.yaw])
        goal = np.array([goal_state.x, goal_state.y, goal_state.yaw])
        self._worker = _GlobalPlannerNodeWorker(
            start, goal, obstacles, segment_collection_size=self._segment_collection_size, parent=self
        )
        self._worker.result.connect(self._on_worker_result)
        self._worker.display_segments.connect(self.display_segments.emit)
        self._worker.start()

    @Slot()
    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()

    @Slot(np.ndarray, bool)
    def _on_worker_result(self, trajectory: Optional[npt.NDArray[np.floating[Any]]], interrupted: bool) -> None:
        if not interrupted:
            self.trajectory.emit(trajectory)
