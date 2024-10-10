import os

os.environ["QT_API"] = "PySide6"


from collections import deque
from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Any, Optional, override

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import QMainWindow

from .CarSimulationNode import CarSimulationNode
from .GlobalPlannerNode import GlobalPlannerNode
from .LocalPlannerNode import LocalPlannerNode
from .modeling.Car import Car
from .modeling.Obstacles import Obstacles
from .ui.mainwindow_ui import Ui_MainWindow

GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE = 50
DASHBOARD_HISTORY_SIZE = 100
DELTA_TIME = 0.1
SIMULATION_INTERVAL = 0.2
CANVAS_ANIMATION_INTERVAL = 0.2
DASHBOARD_ANIMATION_INTERVAL = 0.2


def _get_test_obstacles() -> Obstacles:
    ox, oy = [], []
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)
    for _ in range(20):
        ox.append(np.random.uniform(0, 60))
        oy.append(np.random.uniform(0, 60))

    return Obstacles(np.vstack((ox, oy)).T)


def _get_random_car(obstacles: Obstacles) -> Car:
    state = np.random.uniform((5, 5, -np.pi), (55, 55, np.pi))
    while Car(*state).check_collision(obstacles):
        state = np.random.uniform((5, 5, -np.pi), (55, 55, np.pi))
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

        # setup ui
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self._visualization_canvas = FigureCanvas(Figure())
        self._dashboard_canvas = FigureCanvas(Figure())
        self._ui.visualization_canvas_layout.addWidget(self._visualization_canvas)
        self._ui.dashboard_canvas_layout.addWidget(self._dashboard_canvas)
        self._navigation_toolbar = NavigationToolbar(self._visualization_canvas)
        self._visualization_ax = self._visualization_canvas.figure.subplots()
        self._visualization_ax.set_aspect("equal", adjustable="datalim")
        self._visualization_ax.grid(True, which="both")
        self._measured_velocity_ax, self._measured_steer_ax = self._dashboard_canvas.figure.subplots(2, 1)
        self._measured_velocity_ax: plt.Axes
        self._measured_steer_ax: plt.Axes
        self._measured_velocity_ax.grid(True)
        self._measured_steer_ax.grid(True)
        self._measured_velocities_artist = self._measured_velocity_ax.plot([], [], "-b")[0]
        self._measured_velocity_ax.set_ylim(Car.MIN_SPEED * 3.6 - 5.0, Car.MAX_SPEED * 3.6 + 5.0)
        self._measured_steers_artist = self._measured_steer_ax.plot([], [], "-r")[0]
        self._measured_steer_ax.set_ylim(-np.rad2deg(Car.MAX_STEER) - 5.0, np.rad2deg(Car.MAX_STEER) + 5.0)
        self._measured_state_artists = self._measured_state.plot(ax=self._visualization_ax)
        self._obstacles_artists = self._visualization_ax.plot(*self._obstacles.coordinates.T, ".r")

        # connect ui signals
        self._ui.cancel_button.clicked.connect(self.cancel)
        self._ui.navigate_button.toggled.connect(lambda _: self._navigation_toolbar.pan())
        self._pressed_position: Optional[_PressedPosition] = None
        self._pressed_position_artists: list[plt.Line2D] = []
        self._visualization_canvas.mpl_connect("button_press_event", self._canvas_on_press)
        self._visualization_canvas.mpl_connect("motion_notify_event", self._canvas_on_move)
        self._visualization_canvas.mpl_connect("button_release_event", self._canvas_on_release)

        # Car Simulation Node
        self._car_simulation_node = CarSimulationNode(
            initial_state=self._measured_state.copy(),
            control=(0.0, 0.0),
            delta_time_s=DELTA_TIME,
            simulation_interval_s=SIMULATION_INTERVAL,
        )
        self._car_simulation_thread = QThread(self)
        self._car_simulation_node.moveToThread(self._car_simulation_thread)
        self._car_simulation_node.measured_state.connect(self._update_measured_state)
        self._car_simulation_thread.started.connect(self._car_simulation_node.start_simulation)
        self.canceled.connect(self._car_simulation_node.stop)
        self.set_state.connect(self._car_simulation_node.set_state)
        self._car_simulation_stopped = True

        # Global Planner Node
        self._global_planner_node = GlobalPlannerNode(segment_collection_size=GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE)
        self._global_planner_thread = QThread(self)
        self._global_planner_node.moveToThread(self._global_planner_thread)
        self._global_planner_node.trajectory.connect(self._update_trajectory)
        self._global_planner_node.display_segments.connect(self._update_global_planner_display_segments)
        self._global_planner_node.finished.connect(self._car_simulation_node.resume)
        self.set_goal.connect(self._global_planner_node.plan)
        self.canceled.connect(self._global_planner_node.cancel)
        self._global_planner_segments_collections: list[LineCollection] = []
        self._trajectory_artist: plt.Line2D = self._visualization_ax.plot([], [], "-b")[0]
        self._goal_unreachable_text: plt.Text = self._visualization_ax.text(
            0, 0, "Goal is unreachable", fontsize=12, color="r", visible=False
        )

        # Local Planner Node
        self._local_planner_node = LocalPlannerNode(delta_time_s=DELTA_TIME)
        self._local_planner_thread = QThread(self)
        self._local_planner_node.moveToThread(self._local_planner_thread)
        self._global_planner_node.trajectory.connect(self._local_planner_node.set_trajectory)
        self._car_simulation_node.measured_state.connect(self._local_planner_node.update)
        self._local_planner_node.control.connect(self._car_simulation_node.set_control)
        self._local_planner_node.local_trajectory.connect(self._update_local_trajectory)
        self._local_planner_node.reference_points.connect(self._update_reference_points)
        self.canceled.connect(self._local_planner_node.cancel)
        self._local_trajectory_artist: plt.Line2D = self._visualization_ax.plot([], [], "-g")[0]
        self._reference_points_artist: plt.Line2D = self._visualization_ax.plot([], [], "xr")[0]

        # start tasks and threads
        self._ax_func_animation = FuncAnimation(
            self._visualization_canvas.figure,
            self._update_visualization_figure,
            interval=int(CANVAS_ANIMATION_INTERVAL * 1000),
            save_count=0,
        )
        self._dashboard_ax_func_animation = FuncAnimation(
            self._dashboard_canvas.figure,
            self._update_dashboard_figure,
            interval=int(DASHBOARD_ANIMATION_INTERVAL * 1000),
            save_count=0,
        )
        self._car_simulation_thread.start()
        self._global_planner_thread.start()
        self._local_planner_thread.start()

    @override
    def closeEvent(self, event) -> None:
        self.cancel()
        super().closeEvent(event)

    @Slot()
    def cancel(self):
        self._car_simulation_stopped = True
        self.canceled.emit()
        self._clear_artists(self._global_planner_segments_collections)
        self._clear_artists(self._local_trajectory_artist)
        self._clear_artists(self._reference_points_artist)
        self._goal_unreachable_text.set_visible(False)

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
        state.plot(artists=self._measured_state_artists)

    @Slot(np.ndarray)
    def _update_trajectory(self, trajectory: Optional[np.ndarray]) -> None:
        self._clear_artists(self._global_planner_segments_collections)
        if trajectory is not None:
            self._trajectory_artist.set_data(*trajectory.T[:2])
            self._car_simulation_stopped = False
        else:
            self._goal_unreachable_text.set_visible(True)

    @Slot(LineCollection)
    def _update_global_planner_display_segments(self, display_segments: LineCollection) -> None:
        artist = self._visualization_ax.add_collection(display_segments)
        self._global_planner_segments_collections.append(artist)

    @Slot(np.ndarray)
    def _update_local_trajectory(self, local_trajectory: npt.NDArray[np.floating[Any]]) -> None:
        if self._car_simulation_stopped:
            return
        self._local_trajectory_artist.set_data(*local_trajectory.T)

    @Slot(np.ndarray)
    def _update_reference_points(self, reference_points: npt.NDArray[np.floating[Any]]) -> None:
        if self._car_simulation_stopped:
            return
        self._reference_points_artist.set_data(*reference_points.T)

    def _update_visualization_figure(self, *_) -> None:
        self._visualization_ax.set_title(f"Timestamp: {self._measured_timestamp:.1f}s")

    def _update_dashboard_figure(self, *_) -> None:
        state = self._measured_state
        xlim = (self._measured_timestamps[0], max(self._measured_timestamps[-1], DASHBOARD_HISTORY_SIZE * DELTA_TIME))
        self._measured_velocities_artist.set_data(self._measured_timestamps, self._measured_velocities)
        self._measured_velocity_ax.set_xlim(*xlim)
        self._measured_velocity_ax.set_title(f"Velocity {state.velocity*3.6:.1f}km/h")

        self._measured_steers_artist.set_data(self._measured_timestamps, self._measured_steers)
        self._measured_steer_ax.set_xlim(*xlim)
        self._measured_steer_ax.set_title(f"Steer {np.rad2deg(state.steer):.1f}°")

    @staticmethod
    def _clear_artists(artists: None | plt.Line2D | MutableSequence[plt.Artist]) -> None:
        if artists is None:
            return
        if isinstance(artists, plt.Line2D):
            artists.set_data([], [])
            return
        for artist in artists:
            artist.remove()
        artists.clear()

    def _canvas_on_press(self, event: MouseEvent) -> None:
        if event.xdata is None:
            return
        if not (self._ui.set_pose_button.isChecked() or self._ui.set_goal_button.isChecked()):
            return
        if not (0 <= event.xdata <= 60 and 0 <= event.ydata <= 60):
            return
        self._pressed_position = _PressedPosition(event.xdata, event.ydata)

    def _canvas_on_move(self, event: MouseEvent) -> None:
        if event.xdata is None or self._pressed_position is None:
            return
        if not (self._ui.set_pose_button.isChecked() or self._ui.set_goal_button.isChecked()):
            return
        if self._pressed_position.yaw is None:
            self._clear_artists(self._pressed_position_artists)
        x, y = self._pressed_position.x, self._pressed_position.y
        self._pressed_position.yaw = np.arctan2(event.ydata - y, event.xdata - x)
        if not self._pressed_position_artists:
            color = "r" if self._ui.set_pose_button.isChecked() else "g"
            self._pressed_position_artists = Car(x, y, self._pressed_position.yaw).plot(
                color=color, ax=self._visualization_ax
            )
        else:
            Car(x, y, self._pressed_position.yaw).plot(artists=self._pressed_position_artists)
        self._visualization_canvas.draw()

    def _canvas_on_release(self, _: MouseEvent) -> None:
        if self._pressed_position is None or self._pressed_position.yaw is None:
            self._pressed_position = None
            return
        if not (self._ui.set_pose_button.isChecked() or self._ui.set_goal_button.isChecked()):
            return

        self.cancel()
        self._clear_artists(self._trajectory_artist)

        state = Car(self._pressed_position.x, self._pressed_position.y, self._pressed_position.yaw)
        if self._ui.set_pose_button.isChecked():
            self.set_state.emit(state)
            self._clear_artists(self._pressed_position_artists)
        elif self._ui.set_goal_button.isChecked():
            self._clear_artists(self._trajectory_artist)
            self.set_goal.emit(self._measured_state, state, self._obstacles)
            self._goal_unreachable_text.set_position((state.x, state.y))

        self._pressed_position = None
        self._visualization_canvas.draw()
