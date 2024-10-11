from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from .modeling.Car import Car


class CarSimulationNode(QObject):

    measured_state = Signal(float, Car)

    def __init__(
        self,
        initial_state: Car,
        delta_time_s: float,
        simulation_interval_s: float,
        simulation_publish_interval_s: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._real_state = initial_state
        self._control_sequence: Optional[npt.NDArray[np.floating[Any]]] = None  # [[timestamp, velocity, steer], ...]
        self._delta_time_s = delta_time_s
        self._timestamp_s = 0.0
        self._stopped = True

        self._simulation_timer = QTimer(self)
        self._simulation_timer.timeout.connect(self._simulate)
        self._simulation_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._simulation_timer.setInterval(int(simulation_interval_s * 1000))

        self._simulation_publish_timer = QTimer(self)
        self._simulation_publish_timer.timeout.connect(self._publish_state)
        self._simulation_publish_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._simulation_publish_timer.setInterval(int(simulation_publish_interval_s * 1000))

    @Slot()
    def _simulate(self):
        self._timestamp_s += self._delta_time_s
        if self._control_sequence is None:
            self._real_state.update(self._delta_time_s)
            return
        t = np.clip(self._timestamp_s, self._control_sequence[0, 0], self._control_sequence[-1, 0])
        timestamps, velocities, steers = self._control_sequence.T
        velocity = np.interp(t, timestamps, velocities)
        steer = np.interp(t, timestamps, steers)
        self._real_state.update_with_control(velocity, steer, self._delta_time_s)

    @Slot()
    def _publish_state(self):
        self.measured_state.emit(self._timestamp_s, self._real_state.copy())

    @Slot()
    def start(self):
        self._simulation_timer.start()
        self._simulation_publish_timer.start()

    @Slot(np.ndarray)
    def set_control_sequence(self, control_sequence: npt.NDArray[Any]) -> None:
        if self._stopped:
            return
        self._control_sequence = control_sequence

    @Slot(Car)
    def set_state(self, state: Car) -> None:
        self._real_state = state.copy()

    @Slot()
    def stop(self) -> None:
        self._real_state.velocity = self._real_state.steer = 0.0
        self._control_sequence = None
        self._stopped = True

    @Slot()
    def resume(self) -> None:
        self._stopped = False
