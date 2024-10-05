from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from global_planner.hybrid_a_star import hybrid_a_star
from local_planner.ModelPredictiveControl import ModelPredictiveControl
from modeling.Car import Car
from modeling.Obstacles import Obstacles

DT = 0.1

START = np.array([10.0, 10.0, np.deg2rad(90.0)])
GOAL = np.array([50.0, 55.0, np.deg2rad(-90.0)])

A_STAR_RESULT_FILE = Path("tmp/trajectory.npy")
A_STAR_READ_FROM_FILE = True


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
    return Obstacles(np.vstack((ox, oy)).T)


def main():
    obstacles = get_test_obstacles()
    if A_STAR_READ_FROM_FILE and A_STAR_RESULT_FILE.exists():
        trajectory = np.load(A_STAR_RESULT_FILE)
    else:
        print("Start Hybrid A* planning")
        trajectory = hybrid_a_star(START, GOAL, obstacles)
        print("Hybrid A* planning Finished")
        A_STAR_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(A_STAR_RESULT_FILE, trajectory)

    _, ax = plt.subplots()
    ax.plot(*obstacles.coordinates.T, ".r")
    ax.plot(*trajectory[:, :2].T, "-b")
    ax.set_aspect("equal")

    print("Start MPC")
    state = Car(*START)
    mpc = ModelPredictiveControl(trajectory.copy())
    t = 0.0
    while True:
        result = mpc.update(state, DT)
        acceleration, steer = result.controls[1]
        state.update_with_control(state.velocity + acceleration * DT, steer, DT)
        print(state)

        t += DT
        artists = []
        artists.extend(state.plot(ax))
        artists.extend(ax.plot(*trajectory[:, :2].T, "-b"))
        artists.extend(ax.plot(*result.states[:, :2].T, "xg"))
        artists.extend(ax.plot(*result.ref_states[:, :2].T, "xr"))
        ax.title.set_text(
            f"Time: {t:.1f}s Velocity: {state.velocity*3.6:.1f}km/s Steer: {np.rad2deg(steer):.1f}° Goal: {result.goal_reached}"
        )
        if result.goal_reached:
            break

        plt.pause(0.1)
        for artist in artists:
            artist.remove()
    print("MPC Finished")
    plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    main()
