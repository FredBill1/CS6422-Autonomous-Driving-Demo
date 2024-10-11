from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from .modeling.Car import Car


class CarSimulationNode(QObject):

    measured_state = Signal(float, Car)

    def __init__(
        self,
        initial_state: Car,
        control: tuple[float, float],
        delta_time_s: float,
        simulation_interval_s: float,
        simulation_publish_interval_s: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._real_state = initial_state
        self._control = control
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
        self._real_state.update_with_control(*self._control, self._delta_time_s)
        self._timestamp_s += self._delta_time_s

    @Slot()
    def _publish_state(self):
        self.measured_state.emit(self._timestamp_s, self._real_state.copy())

    @Slot()
    def start(self):
        self._simulation_timer.start()
        self._simulation_publish_timer.start()

    @Slot(tuple)
    def set_control(self, control: tuple[float, float]) -> None:
        if self._stopped:
            return
        self._control = control

    @Slot(Car)
    def set_state(self, state: Car) -> None:
        self._real_state = state.copy()

    @Slot()
    def stop(self) -> None:
        self._real_state.velocity = self._real_state.steer = 0.0
        self._control = (0.0, 0.0)
        self._stopped = True

    @Slot()
    def resume(self) -> None:
        self._stopped = False
