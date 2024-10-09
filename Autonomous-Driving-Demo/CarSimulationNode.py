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
        simulation_delta_time_s: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._real_state = initial_state
        self._control = control
        self._delta_time_s = delta_time_s
        self._simulation_delta_time_s = simulation_delta_time_s
        self._timestamp_s = 0.0

    @Slot()
    def _simulate(self):
        self._real_state.update_with_control(*self._control, self._delta_time_s)
        self._timestamp_s += self._delta_time_s
        self.measured_state.emit(self._timestamp_s, self._real_state.copy())

    @Slot()
    def start_simulation(self):
        self._simulation_timer = QTimer(self)
        self._simulation_timer.timeout.connect(self._simulate)
        self._simulation_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._simulation_timer.start(int(self._simulation_delta_time_s * 1000))

    @Slot(tuple)
    def set_control(self, control: tuple[float, float]) -> None:
        self._control = control

    @Slot(Car)
    def set_state(self, state: Car) -> None:
        self._real_state = state.copy()

    @Slot()
    def cancel(self):
        self._real_state.velocity = self._real_state.steer = 0.0
        self._control = (0.0, 0.0)
