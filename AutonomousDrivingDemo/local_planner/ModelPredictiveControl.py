from typing import Any, NamedTuple

import cvxpy
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.optimize

from ..modeling.Car import Car
from ..utils.wrap_angle import smooth_yaw, wrap_angle

COURSE_TICK = 0.5  # [m], equal to MOTION_RESOLUTION from hybrid_a_star.py

NEARIST_POINT_SEARCH_RANGE = 3.0  # [m]
NEARIST_POINT_SEARCH_STEP = 0.1  # [m]

HORIZON_LENGTH = 5  # simulate count
MIN_HORIZON_DISTANCE = 2.0  # [m]

MAX_ITER = 5
DU_TH = 0.1  # iteration finish param


# mpc parameters
R = np.diag([0.01, 0.005])  # input cost matrix
R_D = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.1, 1.1, 0.05, 1.1])  # state cost matrix
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

    Reference:
    https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
    """
    # Note: ndarrays in this function are transposed
    x = cvxpy.Variable((NX, HORIZON_LENGTH + 1))  # [x, y, v, yaw]
    u = cvxpy.Variable((NU, HORIZON_LENGTH))  # [accel, steer]

    cost = 0.0
    constraints = []
    for t in range(HORIZON_LENGTH):
        # prefer control to be small
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            # prefer state to be close to reference
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        # make sure the next state is transformed from the previous state using the approximated linear model
        A, B, C = _get_linear_model_matrix(xbar[2, t], xbar[3, t], last_steer, dt)
        constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C)

    # prefer final state to be close to reference
    cost += cvxpy.quad_form(xref[:, HORIZON_LENGTH] - x[:, HORIZON_LENGTH], Q_F)

    # make sure steer change is within the maximum steer speed
    constraints.append(cvxpy.abs(u[1, 0] - last_steer) <= Car.MAX_STEER_SPEED * dt)

    for t in range(1, HORIZON_LENGTH):
        # prefer control difference between two steps to be small
        cost += cvxpy.quad_form(u[:, t] - u[:, t - 1], R_D)

        # make sure steer change is within the maximum steer speed
        constraints.append(cvxpy.abs(u[1, t] - u[1, t - 1]) <= Car.MAX_STEER_SPEED * dt)

    # make sure initial state is the current state
    constraints.append(x[:, 0] == xbar[:, 0])
    # make sure speed at every step is in range
    constraints.append(x[2, :] <= Car.MAX_SPEED)
    constraints.append(x[2, :] >= Car.MIN_SPEED)
    # make sure acceleration at every step is in range
    constraints.append(cvxpy.abs(u[0, :]) <= Car.MAX_ACCEL)
    constraints.append(cvxpy.abs(u[1, :]) <= Car.MAX_STEER)

    # solve the problem
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
        ref_trajectory[:, 2] = smooth_yaw(ref_trajectory[:, 2])
        v = ref_trajectory[:, 3]
        v[:-1][v[1:] != v[:-1]] = 0.0  # make the vehicle stop at the direction change point
        v *= Car.TARGET_SPEED  # make the target velocity at each point of the trajectory to be TARGET_SPEED
        v[-1] = 0.0  # make the vehicle stop at the goal
        self._goal = ref_trajectory[-1]

        # [x, y, yaw, v] -> [x, y, v, yaw]
        ref_trajectory = ref_trajectory[:, [0, 1, 3, 2]]

        # interpolate the reference trajectory
        dists = np.linalg.norm(ref_trajectory[1:, :2] - ref_trajectory[:-1, :2], axis=1)
        u = np.concatenate(([0], np.cumsum(dists)))
        self._tck, self._u = scipy.interpolate.splprep(ref_trajectory.T, s=0, k=1, u=u)
        self._cur_u = 0.0

    def _nearist_point(self, state: Car) -> float:
        "find the nearist point on the reference trajectory to the given state"

        def cost(u: float) -> float:
            return np.linalg.norm(np.array(scipy.interpolate.splev(u, self._tck)[:2]).T - [state.x, state.y])

        min_dist, min_u = np.inf, self._cur_u
        for u in np.arange(self._cur_u, self._cur_u + NEARIST_POINT_SEARCH_RANGE, NEARIST_POINT_SEARCH_STEP):
            if (dist := cost(u)) < min_dist:
                min_dist, min_u = dist, u
            else:
                break
        return min_u

    def _goal_reached(self, state: Car) -> bool:
        return (
            np.linalg.norm(self._goal[:2] - [state.x, state.y]) < GOAL_MAX_DISTANCE
            and abs(wrap_angle(self._goal[2] - state.yaw)) < GOAL_YAW_DIFF
            and abs(state.velocity) < GOAL_MAX_SPEED
        )

    def _find_xref(self, state: Car, dt: float) -> npt.NDArray[np.floating[Any]]:
        "Find the closest point in the reference trajectory, and interpolate the reference trajectory within a horizon"
        while True:
            # interpolate the reference trajectory
            self._cur_u = self._nearist_point(state)
            length = max(MIN_HORIZON_DISTANCE, abs(state.velocity) * dt * HORIZON_LENGTH)
            ref_u = np.linspace(self._cur_u, self._cur_u + length, HORIZON_LENGTH + 1)
            ref_u = np.clip(ref_u, a_min=None, a_max=self._u[-1])
            xref = np.array(scipy.interpolate.splev(ref_u, self._tck)).T

            # check if the reference trajectory contains a direction change
            v = xref[:, 2]
            for i in range(1, len(v)):
                if v[i] * v[i - 1] < 0:
                    break
            else:
                return xref  # if not, return the reference trajectory

            # use binary search to find the direction change point:
            l, r = ref_u[i - 1], ref_u[i]
            while r - l > 1e-6:
                m = (l + r) / 2
                if xref[0, 2] * scipy.interpolate.splev(m, self._tck)[2] < 0:
                    r = m
                else:
                    l = m

            # if the direction change happens only after the first points, we discard the first point and start to track the
            # trajectory from the direction change point
            if i <= 1:
                self._cur_u = r
                continue

            # otherwise, we make the direction change point to have zero velocity, and the vehicle should stop at that point
            xref = xref[:i]
            xref = np.vstack([xref, np.array(scipy.interpolate.splev(r, self._tck)).T])
            xref = np.pad(xref, ((0, HORIZON_LENGTH + 1 - len(xref)), (0, 0)), mode="edge")
            xref[i:, 2] = xref[0, 2]
            xref[-1, 2] = 0.0
            return xref

    def update(self, state: Car, dt: float) -> MPCResult:
        xref = self._find_xref(state, dt)

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
            print("Warning: Cannot converge mpc")
        return MPCResult(controls, states[:, [0, 1, 3, 2]], xref[:, [0, 1, 3, 2]], self._goal_reached(state))
