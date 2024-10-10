from typing import Any, NamedTuple

import cvxpy
import numpy as np
import numpy.typing as npt

from ..modeling.Car import Car
from ..utils.wrap_angle import smooth_yaw, wrap_angle

COURSE_TICK = 0.5  # [m], equal to MOTION_RESOLUTION from hybrid_a_star.py

NEARIST_POINT_SEARCH_RANGE = 3.0  # [m]
NEARIST_POINT_SEARCH_COUNT = int(NEARIST_POINT_SEARCH_RANGE // COURSE_TICK)

HORIZON_LENGTH = 6  # simulate count
MIN_HORIZON_DISTANCE = 2.0  # [m]

MAX_ITER = 5
DU_TH = 0.1  # iteration finish param


# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
R_D = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 1.1])  # state cost matrix
Q_F = Q * 2  # state final matrix

NX = 4  # [x, y, v, yaw]
NU = 2  # [accel, steer]

GOAL_MAX_DISTANCE = 1.5  # [m]
GOAL_YAW_DIFF = np.deg2rad(45.0)  # [rad]
GOAL_MAX_SPEED = 1.0 / 3.6  # [m/s]


def _get_linear_model_matrix(
    velocity: float, yaw: float, steer: float, dt: float
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    The linear approximation of the vehicle's motion model, by assuming velocity, yaw, and steer are constant in dt.

    >>> X[t+dt] = A @ X[t] + B @ u[t] + C
    where X[t] = [x, y, v, yaw], u[t] = [accel, steer] of timestamp t.
    """
    # Note: ndarrays in this function are transposed
    sy, cy, cs = np.sin(yaw), np.cos(yaw), np.cos(steer)
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = dt * cy
    A[0, 3] = -dt * velocity * sy
    A[1, 2] = dt * sy
    A[1, 3] = dt * velocity * cy
    A[3, 2] = dt * np.tan(steer) / Car.WHEEL_BASE
    B = np.zeros((NX, NU))
    B[2, 0] = dt
    B[3, 1] = dt * velocity / (Car.WHEEL_BASE * cs**2)
    C = np.zeros(NX)
    C[0] = dt * velocity * sy * yaw
    C[1] = -dt * velocity * cy * yaw
    C[3] = -dt * velocity * steer / (Car.WHEEL_BASE * cs**2)

    return A, B, C


def _predict_motion(state: Car, controls: npt.NDArray[np.floating[Any]], dt: float) -> npt.NDArray[np.floating[Any]]:
    """
    Predict the next `HORIZON_LENGTH` motions of the vehicle with the given controls, `len(controls) == HORIZON_LENGTH`.
    """
    state = state.copy()
    states = [[state.x, state.y, state.velocity, state.yaw]]
    for acceleration, steer in controls:
        state.update_with_control(state.velocity + acceleration * dt, steer, dt, do_wrap_angle=False)
        states.append([state.x, state.y, state.velocity, state.yaw])
    return np.array(states)


def _linear_mpc_control(
    xref: npt.NDArray[np.floating[Any]], xbar: npt.NDArray[np.floating[Any]], last_steer: float, dt: float
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    The car motion model is approximated as linear on dt, and cvxpy is used to solve the optimal control
    output u after approximation. Recalculate u as the new initial condition, and it will converge to a
    local minimum after several iterations.
    """

    # Note: ndarrays in this function are transposed
    "https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py"
    x = cvxpy.Variable((NX, HORIZON_LENGTH + 1))  # [x, y, v, yaw]
    u = cvxpy.Variable((NU, HORIZON_LENGTH))  # [accel, steer]

    cost = 0.0
    constraints = []
    for t in range(HORIZON_LENGTH):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)
        A, B, C = _get_linear_model_matrix(xbar[2, t], xbar[3, t], last_steer, dt)
        constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C)
    cost += cvxpy.quad_form(xref[:, HORIZON_LENGTH] - x[:, HORIZON_LENGTH], Q_F)

    constraints.append(cvxpy.abs(u[1, 0] - last_steer) <= Car.MAX_STEER_SPEED * dt)
    for t in range(1, HORIZON_LENGTH):
        cost += cvxpy.quad_form(u[:, t] - u[:, t - 1], R_D)
        constraints.append(cvxpy.abs(u[1, t] - u[1, t - 1]) <= Car.MAX_STEER_SPEED * dt)

    constraints.append(x[:, 0] == xbar[:, 0])
    constraints.append(x[2, :] <= Car.MAX_SPEED)
    constraints.append(x[2, :] >= Car.MIN_SPEED)
    constraints.append(cvxpy.abs(u[0, :]) <= Car.MAX_ACCEL)
    constraints.append(cvxpy.abs(u[1, :]) <= Car.MAX_STEER)

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        print(f"Error: Cannot solve mpc: {prob.status}")
        return None
    return u.value, x.value


class MPCResult(NamedTuple):
    controls: npt.NDArray[np.floating[Any]]  # [[accel, steer]], target output controls
    states: npt.NDArray[np.floating[Any]]  # [[x, y, v, yaw]], predicted states
    ref_states: npt.NDArray[np.floating[Any]]  # [[x, y, v, yaw]], reference states on the trajectory
    goal_reached: bool  # whether the goal is reached


class ModelPredictiveControl:
    def __init__(self, ref_trajectory: npt.NDArray[np.floating[Any]]) -> None:
        assert ref_trajectory.shape[1] == 4, "Reference trajectory have [[x, y, yaw, direction], ...]"
        self._ref_trajectory = ref_trajectory
        self._ref_trajectory[:, 2] = smooth_yaw(self._ref_trajectory[:, 2])
        self._ref_trajectory[:, 3] *= Car.TARGET_SPEED
        N = self._ref_trajectory.shape[0]
        for i in range(min(N, round(GOAL_MAX_DISTANCE / COURSE_TICK))):
            self._ref_trajectory[N - 1 - i, 3] = 0.0

    def _nearist_point_index(self, state: Car) -> int:
        ref_xy = self._ref_trajectory[:NEARIST_POINT_SEARCH_COUNT, :2]
        dists = np.linalg.norm(ref_xy - [state.x, state.y], axis=1)
        return np.argmin(dists)

    def _goal_reached(self, state: Car) -> bool:
        return (
            np.linalg.norm(self._ref_trajectory[-1, :2] - [state.x, state.y]) < GOAL_MAX_DISTANCE
            and abs(wrap_angle(self._ref_trajectory[-1, 2] - state.yaw)) < GOAL_YAW_DIFF
            and abs(state.velocity) < GOAL_MAX_SPEED
        )

    def update(self, state: Car, dt: float) -> MPCResult:
        # Find the closest point in the reference trajectory
        self._ref_trajectory = self._ref_trajectory[self._nearist_point_index(state) :]
        ids = np.round(
            np.arange(HORIZON_LENGTH + 1)
            * (max(MIN_HORIZON_DISTANCE / HORIZON_LENGTH, abs(state.velocity) * dt) / COURSE_TICK)
        )
        xref = self._ref_trajectory[np.clip(ids.astype(int), None, self._ref_trajectory.shape[0] - 1)]
        xref = xref[:, [0, 1, 3, 2]]  # [x, y, yaw, v] -> [x, y, v, yaw]

        # Align the yaw of the vehicle with the reference trajectory, to facilitate the calculation of
        # the yaw difference between the current state and the reference trajectory.
        state = state.copy()
        state.align_yaw(xref[0, 3])

        # iteratively solve the linearized problem
        controls, states = np.zeros((HORIZON_LENGTH, NU)), np.zeros((HORIZON_LENGTH + 1, NX))
        for _ in range(MAX_ITER):
            xbar = _predict_motion(state, controls, dt)
            pre_controls = controls.copy()
            res = _linear_mpc_control(xref.T, xbar.T, state.steer, dt)
            if res is None:
                break
            controls, states = res[0].T, res[1].T
            du = np.linalg.norm(controls - pre_controls)
            if du < DU_TH:
                break
        else:
            print("Error: Cannot converge mpc")
        return MPCResult(controls, states[:, [0, 1, 3, 2]], xref[:, [0, 1, 3, 2]], self._goal_reached(state))
