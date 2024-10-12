import PySide6

"set PySide6 backend"

from collections import deque
from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Any, Optional, override

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import QMainWindow

from .CarSimulationNode import CarSimulationNode
from .GlobalPlannerNode import GlobalPlannerNode
from .LocalPlannerNode import LocalPlannerNode
from .modeling.Car import Car
from .modeling.Obstacles import Obstacles
from .plotting.CarItem import CarItem
from .ui.mainwindow_ui import Ui_MainWindow

GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE = 50

SIMULATION_DELTA_TIME = 0.01
SIMULATION_INTERVAL = 0.02
SIMULATION_PUBLISH_INTERVAL = 0.1
SIMULATION_PUBLISH_DELTA_TIME = SIMULATION_DELTA_TIME * SIMULATION_PUBLISH_INTERVAL / SIMULATION_INTERVAL

LOCAL_PLANNER_DELTA_TIME = 0.1
LOCAL_PLANNER_UPDATE_INTERVAL = 0.2

CANVAS_ANIMATION_INTERVAL = 0.1
DASHBOARD_ANIMATION_INTERVAL = 1.0
DASHBOARD_HISTORY_SIZE = 250

MAP_WIDTH = 60.0
MAP_HEIGHT = 60.0
MAP_STEP = 1.0
MAP_NUM_RANDOM_OBSTACLES = 40


def _get_test_obstacles() -> Obstacles:
    ox = [
        np.arange(0, MAP_WIDTH, MAP_STEP),
        np.full(np.ceil(MAP_HEIGHT / MAP_STEP).astype(int), MAP_WIDTH),
        np.arange(0, MAP_WIDTH + MAP_STEP, MAP_STEP),
        np.full(np.ceil(MAP_HEIGHT / MAP_STEP).astype(int) + 1, 0.0),
        np.full(np.ceil(MAP_WIDTH / 3 * 2 / MAP_STEP).astype(int), MAP_WIDTH / 3),
        np.full(np.ceil(MAP_HEIGHT / 3 * 2 / MAP_STEP).astype(int), 2 * MAP_WIDTH / 3),
        np.random.uniform(0, MAP_WIDTH, MAP_NUM_RANDOM_OBSTACLES),
    ]
    oy = [
        np.full(np.ceil(MAP_WIDTH / MAP_STEP).astype(int), 0.0),
        np.arange(0, MAP_HEIGHT, MAP_STEP),
        np.full(np.ceil((MAP_WIDTH + MAP_STEP) / MAP_STEP).astype(int), MAP_HEIGHT),
        np.arange(0, MAP_HEIGHT + MAP_STEP, MAP_STEP),
        np.arange(0, MAP_WIDTH / 3 * 2, MAP_STEP),
        MAP_HEIGHT - np.arange(0, MAP_HEIGHT / 3 * 2, MAP_STEP),
        np.random.uniform(0, MAP_HEIGHT, MAP_NUM_RANDOM_OBSTACLES),
    ]
    return Obstacles(np.vstack((np.concatenate(ox), np.concatenate(oy))).T)


def _get_random_car(obstacles: Obstacles) -> Car:
    state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
    while Car(*state).check_collision(obstacles):
        state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
    return Car(*state)


@dataclass(slots=True)
class _PressedPosition:
    x: float
    y: float
    yaw: Optional[float] = None


class MainWindow(QMainWindow):
    set_state = Signal(Car)
    set_goal = Signal(Car, Car, Obstacles)
    canceled = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # prepare data
        self._obstacles = _get_test_obstacles()
        self._measured_state = _get_random_car(self._obstacles)
        self._measured_timestamp = 0.0
        self._measured_velocities: deque[float] = deque([0.0], maxlen=DASHBOARD_HISTORY_SIZE)
        self._measured_steers: deque[float] = deque([0.0], maxlen=DASHBOARD_HISTORY_SIZE)
        self._measured_timestamps: deque[float] = deque([0.0], maxlen=DASHBOARD_HISTORY_SIZE)
        self._car_simulation_stopped = True
        self._pressed_position: Optional[_PressedPosition] = None

        # setup ui
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._plot_widget = pg.PlotWidget()
        self._ui.visualization_canvas_layout.addWidget(self._plot_widget)
        self._plot_widget.setAspectLocked()
        self._plot_widget.addItem(pg.GridItem())
        self._obstacle_item = pg.ScatterPlotItem(
            *self._obstacles.coordinates.T, size=5, symbol="o", pen=None, brush=(255, 0, 0)
        )
        self._plot_widget.addItem(self._obstacle_item)
        self._measured_state_item = CarItem(self._measured_state)
        self._plot_widget.addItem(self._measured_state_item)
