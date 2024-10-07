from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue

import matplotlib.pyplot as plt
import numpy as np

from global_planner.hybrid_a_star import Node, hybrid_a_star
from local_planner.ModelPredictiveControl import ModelPredictiveControl
from localization.estimate_rigid_transform import estimate_rigid_transform
from modeling.Car import Car
from modeling.Obstacles import Obstacles

DT = 0.1

MIN_SCAN_NUM = 10


def get_test_obstacles() -> Obstacles:
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


def main(ax: plt.Axes):
    obstacles = get_test_obstacles()

    def get_random_car(x, y):
        state = np.random.uniform((x - 5, y - 5, -np.pi), (x + 5, y + 5, np.pi))
        while Car(*state).check_collision(obstacles):
            state = np.random.uniform((x - 5, y - 5, -np.pi), (x + 5, y + 5, np.pi))
        return state

    START, GOAL = get_random_car(10, 10), get_random_car(50, 50)
    if np.random.rand() < 0.5:
        START, GOAL = GOAL, START
    print(f"Start: {repr(START)}\nGoal: {repr(GOAL)}")

    print("Start Hybrid A* planning")
    ax.cla()
    ax.plot(*obstacles.coordinates.T, ".r")
    ax.arrow(*START[:2], np.cos(START[2]), np.sin(START[2]), head_width=0.5, head_length=1, fc="r", ec="r")
    ax.arrow(*GOAL[:2], np.cos(GOAL[2]), np.sin(GOAL[2]), head_width=0.5, head_length=1, fc="g", ec="g")
    ax.set_aspect("equal")
    ax.title.set_text("Hybrid A* Planning")
    plt.pause(0.1)

    explored_nodes = SimpleQueue()
    explored_nodes_artists = []
    with ThreadPoolExecutor(1) as executor:
        global_planner_thread = executor.submit(hybrid_a_star, START, GOAL, obstacles, explored_nodes.put)
        while not global_planner_thread.done():
            nodes: list[Node] = []
            while True:
                try:
                    nodes.append(explored_nodes.get_nowait())
                except Empty:
                    break
            for node in nodes:
                explored_nodes_artists.extend(ax.plot(*node.get_plot_trajectory()[:, :2].T, "-y"))
            plt.pause(0.1)
        trajectory = global_planner_thread.result()
    print("Hybrid A* planning Finished")
    if trajectory is None:
        print("Goal is not reachable")
        return

    for artist in explored_nodes_artists:
        artist.remove()

    ax.plot(*trajectory[:, :2].T, "-b")

    print("Start MPC")
    real_state = Car(*START)
    est_state = real_state.copy()
    mpc = ModelPredictiveControl(trajectory.copy())
    result = None
    timestamp = 0.0
    while True:
        artists = []
        artists.extend(real_state.plot(ax, color="-k"))
        artists.extend(est_state.plot(ax, color="-r"))
        if result is not None and result.goal_reached:
            break

        result = mpc.update(est_state, DT)
        acceleration, steer = result.controls[1]
        control = (est_state.velocity + acceleration * DT, steer)
        real_state.update_with_control(*control, DT, with_noise=True)
        est_state.update_with_control(*control, DT, with_noise=False)

        ids, scan = real_state.scan_obstacles(obstacles, with_noise=True)
        artists.extend(ax.plot(*real_state.local_to_world(scan).T, "xg"))
        circle = plt.Circle(
            (real_state.x, real_state.y),
            real_state.SCAN_RADIUS,
            fill=False,
            color="r" if len(scan) >= MIN_SCAN_NUM else "g",
        )
        ax.add_artist(circle)
        artists.append(circle)
        if len(scan) >= MIN_SCAN_NUM:
            ref_scan = obstacles.coordinates[ids]
            R, t = estimate_rigid_transform(scan, ref_scan)
            est_state.x, est_state.y = t
            est_state.yaw = np.arctan2(R[0, 1], R[0, 0])

        timestamp += DT
        artists.extend(ax.plot(*result.states[:, :2].T, "-g"))
        artists.extend(ax.plot(*result.ref_states[:, :2].T, "xr"))
        ax.title.set_text(
            f"Time: {timestamp:.1f}s Velocity: {real_state.velocity*3.6:.1f}km/s Steer: {np.rad2deg(steer):.1f}° Goal: {result.goal_reached}"
        )

        plt.pause(0.1)
        for artist in artists:
            artist.remove()
    print("MPC Finished")
    plt.pause(1)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", lambda _: quit())
    while True:
        main(ax)
