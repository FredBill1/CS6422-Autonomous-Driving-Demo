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
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QMouseEvent
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


class _CustomViewBox(pg.ViewBox):
    sigMouseDrag = Signal(MouseDragEvent)

    def mouseDragEvent(self, ev: MouseDragEvent) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseDragEvent(ev)
        ev.accept()
        self.sigMouseDrag.emit(ev)


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

        # setup ui
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._plot_viewbox = _CustomViewBox()
        self._plot_viewbox.sigMouseDrag.connect(self._mouse_drag)

        self._plot_widget = pg.PlotWidget(viewBox=self._plot_viewbox)
        self._ui.visualization_canvas_layout.addWidget(self._plot_widget)
        self._plot_widget.setAspectLocked()
        self._plot_widget.addItem(pg.GridItem())
        self._obstacle_item = pg.ScatterPlotItem(
            *self._obstacles.coordinates.T, size=5, symbol="o", pen=None, brush=(255, 0, 0)
        )
        self._plot_widget.addItem(self._obstacle_item)
        self._measured_state_item = CarItem(self._measured_state)
        self._plot_widget.addItem(self._measured_state_item)

        self._pressed_pose_item = CarItem(self._measured_state)
        self._pressed_pose_item.setVisible(False)
        self._plot_widget.addItem(self._pressed_pose_item)
        self._goal_pose_item = CarItem(self._measured_state, color="r")
        self._goal_pose_item.setVisible(False)
        self._plot_widget.addItem(self._goal_pose_item)
        self._goal_unreachable_item = pg.TextItem("Goal is unreachable", color="r")
        font = QFont()
        font.setPointSize(15)
        self._goal_unreachable_item.setFont(font)
        self._goal_unreachable_item.setVisible(False)
        self._plot_widget.addItem(self._goal_unreachable_item)

        self._global_planner_node = GlobalPlannerNode(
            segment_collection_size=GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE,
        )
        self._global_planner_segments_items: list[pg.PlotCurveItem] = []
        self._trajectory_item = pg.PlotCurveItem(pen=pg.mkPen("b", width=2))
        self._trajectory_item.setVisible(False)
        self._plot_widget.addItem(self._trajectory_item)

        self._car_simulation_node = CarSimulationNode(
            initial_state=self._measured_state.copy(),
            delta_time_s=SIMULATION_DELTA_TIME,
            simulation_interval_s=SIMULATION_INTERVAL,
            simulation_publish_interval_s=SIMULATION_PUBLISH_INTERVAL,
        )

        self._local_planner_node = LocalPlannerNode(
            delta_time_s=LOCAL_PLANNER_DELTA_TIME,
            update_interval_s=LOCAL_PLANNER_UPDATE_INTERVAL,
        )

        self._car_simulation_node.measured_state.connect(self._local_planner_node.set_state)
        self._car_simulation_node.measured_state.connect(self._update_measured_state)
        self._global_planner_node.display_segments.connect(self._update_global_planner_display_segments)
        self._global_planner_node.finished.connect(self._car_simulation_node.resume)
        self._global_planner_node.trajectory.connect(self._local_planner_node.set_trajectory)
        self._global_planner_node.trajectory.connect(self._update_trajectory)
        self._local_planner_node.control_sequence.connect(self._car_simulation_node.set_control_sequence)
        # self._local_planner_node.local_trajectory.connect(self._update_local_trajectory)
        # self._local_planner_node.reference_points.connect(self._update_reference_points)
        self.canceled.connect(self._car_simulation_node.stop)
        self.canceled.connect(self._global_planner_node.cancel)
        self.canceled.connect(self._local_planner_node.cancel)
        self.set_goal.connect(self._global_planner_node.plan)
        self.set_state.connect(self._car_simulation_node.set_state)

        self._plot_viewbox.disableAutoRange()
        self._car_simulation_node.start()
        self._global_planner_node.start()
        self._local_planner_node.start()

    @Slot()
    def cancel(self):
        self._car_simulation_stopped = True
        self.canceled.emit()
        self._goal_unreachable_item.setVisible(False)

    @Slot(MouseDragEvent)
    def _mouse_drag(self, ev: MouseDragEvent) -> None:
        start_pos = self._plot_viewbox.mapSceneToView(ev.buttonDownScenePos())
        start_x, start_y = start_pos.x(), start_pos.y()
        if not (0 <= start_x <= MAP_WIDTH and 0 <= start_y <= MAP_HEIGHT):
            return

        self._goal_unreachable_item.setVisible(False)

        pos = self._plot_viewbox.mapSceneToView(ev.scenePos())
        x, y = pos.x(), pos.y()
        state = Car(start_x, start_y, np.arctan2(y - start_y, x - start_x))
        self._pressed_pose_item.set_state(state)
        self._pressed_pose_item.setVisible(True)
        if self._ui.set_goal_button.isChecked():
            self._pressed_pose_item.set_color("r")
        elif self._ui.set_pose_button.isChecked():
            self._pressed_pose_item.set_color("g")

        if not ev.isFinish():
            return

        self._pressed_pose_item.setVisible(False)
        self._trajectory_item.setVisible(False)
        self._clear_global_planner_display_segments()

        if self._ui.set_pose_button.isChecked():
            self.set_state.emit(state)
            self._goal_pose_item.setVisible(False)
        elif self._ui.set_goal_button.isChecked():
            self.set_goal.emit(self._measured_state, state, self._obstacles)
            self._goal_pose_item.set_state(state)
            self._goal_pose_item.setVisible(True)
            self._goal_unreachable_item.setPos(start_x, start_y)

    @Slot(list)
    def _update_global_planner_display_segments(self, display_segments: list[npt.NDArray[np.floating[Any]]]) -> None:
        connects = [np.ones(len(segment), dtype=bool) for segment in display_segments]
        for i in range(len(connects) - 1):
            connects[i][-1] = False
        connects = np.concatenate(connects)
        segments = np.vstack(display_segments)
        pen = pg.mkPen(len(self._global_planner_segments_items), 16)
        item = pg.PlotCurveItem(*segments.T, pen=pen, skipFiniteCheck=True, connect=connects)
        self._plot_widget.addItem(item)
        self._global_planner_segments_items.append(item)

    def _clear_global_planner_display_segments(self) -> None:
        for item in self._global_planner_segments_items:
            self._plot_widget.removeItem(item)
        self._global_planner_segments_items.clear()

    @Slot(np.ndarray)
    def _update_trajectory(self, trajectory: Optional[np.ndarray]) -> None:
        self._clear_global_planner_display_segments()

        if trajectory is not None:
            self._trajectory_item.setData(*trajectory.T[:2])
            self._trajectory_item.setVisible(True)
            self._car_simulation_stopped = False
        else:
            self._goal_unreachable_item.setVisible(True)

    @Slot(float, Car)
    def _update_measured_state(self, timestamp_s: float, state: Car) -> None:
        self._measured_state = state
        self._measured_timestamp = timestamp_s
        if len(self._measured_timestamps) >= DASHBOARD_HISTORY_SIZE:
            self._measured_timestamps.popleft()
            self._measured_velocities.popleft()
            self._measured_steers.popleft()
        self._measured_timestamps.append(timestamp_s)
        self._measured_velocities.append(state.velocity * 3.6)  # m/s -> km/h
        self._measured_steers.append(np.rad2deg(state.steer))
        self._measured_state_item.set_state(state)
