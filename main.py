import matplotlib.pyplot as plt
import numpy as np

from global_planner.hybrid_a_star import hybrid_a_star
from local_planner.ModelPredictiveControl import ModelPredictiveControl
from modeling.Car import Car
from modeling.Obstacles import Obstacles

DT = 0.1


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


def main(ax: plt.Axes):
    obstacles = get_test_obstacles()

    def get_random_car():
        x = np.random.uniform((5, 5, -np.pi), (55, 55, np.pi))
        while Car(*x).check_collision(obstacles):
            x = np.random.uniform((5, 5, -np.pi), (55, 55, np.pi))
        return x

    START, GOAL = get_random_car(), get_random_car()
    print(f"Start: {repr(START)}\nGoal: {repr(GOAL)}")

    print("Start Hybrid A* planning")
    trajectory = hybrid_a_star(START, GOAL, obstacles)
    print("Hybrid A* planning Finished")
    if trajectory is None:
        print("Goal is not reachable")
        return

    ax.cla()
    ax.plot(*obstacles.coordinates.T, ".r")
    ax.plot(*trajectory[:, :2].T, "-b")
    ax.set_aspect("equal")

    print("Start MPC")
    state = Car(*START)
    mpc = ModelPredictiveControl(trajectory.copy())
    result = None
    t = 0.0
    while True:
        artists = []
        artists.extend(state.plot(ax))
        if result is not None and result.goal_reached:
            break

        result = mpc.update(state, DT)
        acceleration, steer = result.controls[1]
        state.update_with_control(state.velocity + acceleration * DT, steer, DT)
        t += DT
        artists.extend(ax.plot(*trajectory[:, :2].T, "-b"))
        artists.extend(ax.plot(*result.states[:, :2].T, "-g"))
        artists.extend(ax.plot(*result.ref_states[:, :2].T, "xr"))
        ax.title.set_text(
            f"Time: {t:.1f}s Velocity: {state.velocity*3.6:.1f}km/s Steer: {np.rad2deg(steer):.1f}° Goal: {result.goal_reached}"
        )

        plt.pause(0.1)
        for artist in artists:
            artist.remove()
    print("MPC Finished")
    plt.pause(1)


if __name__ == "__main__":
    _, ax = plt.subplots()
    while True:
        main(ax)
