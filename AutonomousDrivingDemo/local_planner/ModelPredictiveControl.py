from typing import Any, NamedTuple, Optional

import cvxpy
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.optimize

from ..modeling.Car import Car
from ..utils.wrap_angle import smooth_yaw

NEARIST_POINT_SEARCH_RANGE = 20.0  # [m]
NEARIST_POINT_SEARCH_STEP = 0.1  # [m]

HORIZON_LENGTH = 5  # simulate count
MIN_HORIZON_DISTANCE = 0.3  # [m]

MAX_ITER = 5
DU_TH = 0.1  # iteration finish param


# mpc parameters
R = np.diag([0.01, 0.005])  # input cost matrix
R_D = np.diag([1e-5, 0.1])  # input difference cost matrix
Q = np.diag([1.1, 1.1, 0.05, 1.1])  # state cost matrix
Q_F = Q * 2  # state final matrix

NX = 4  # [x, y, v, yaw]
NU = 2  # [accel, steer]

DESIRED_MAX_ACCEL_RATIO = 0.7  # desired max acceleration ratio when the vehicle needs to stop


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
) -> Optional[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
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
        cost += cvxpy.quad_form((u[:, t] - u[:, t - 1]) / dt, R_D)

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


class ModelPredictiveControl:
    def __init__(self, ref_trajectory: npt.NDArray[np.floating[Any]]) -> None:
        assert ref_trajectory.shape[1] == 4, "Reference trajectory have [[x, y, yaw, direction], ...]"
        assert (ref_trajectory[:, 3] != 0).all(), "the direction on each point of the trajectory should not be zero"

        # remove consecutive identical points
        xy = ref_trajectory[:, :2]
        mask = (xy[:-1] != xy[1:]).any(axis=1)
        ref_trajectory = ref_trajectory[np.concatenate(([True], mask))]

        ref_trajectory[:, 2] = smooth_yaw(ref_trajectory[:, 2])

        trajectory = [ref_trajectory[0]]
        for i, point in enumerate(ref_trajectory[1:], 1):
            if (last_point := trajectory[-1])[3] != point[3]:  # last_point is a direction changing point
                assert i > 1, "The first point of the trajectory should not be a direction changing point"
                # set the direction changing point to have zero velocity, and add two points having non-zero velocity on the two sides of it
                trajectory[-1] = (last_point + trajectory[-2]) / 2
                trajectory.append(last_point)
                trajectory.append((point + last_point) / 2)
                trajectory[-2][3], trajectory[-1][3] = 0.0, point[3]
            trajectory.append(point)
        trajectory[-1][3] = 0.0  # make the goal point to have zero velocity
        ref_trajectory = np.vstack(trajectory)
        ref_trajectory[:, 3] *= Car.TARGET_SPEED

        # [x, y, yaw, v] -> [x, y, v, yaw]
        ref_trajectory = ref_trajectory[:, [0, 1, 3, 2]]

        # calculate the ticks of the reference trajectory
        dists = np.linalg.norm(ref_trajectory[1:, :2] - ref_trajectory[:-1, :2], axis=1)
        u = np.concatenate(([0], np.cumsum(dists)))

        # limit the velocity by max acceleration when the vehicle needs to stop
        v = ref_trajectory[:, 2]
        last_zero = None
        for i in reversed(range(len(v))):
            if v[i] == 0:
                last_zero = u[i]
            elif last_zero is not None:
                dist = last_zero - u[i]
                limit = np.sqrt(2 * DESIRED_MAX_ACCEL_RATIO * Car.MAX_ACCEL * dist)
                v[i] = np.clip(v[i], -limit, limit)

        self._direction_changing_us = u[ref_trajectory[:, 2] == 0.0][:-1]

        # interpolate the reference trajectory
        self._tck, us = scipy.interpolate.splprep(ref_trajectory.T, s=0, k=1, u=u)
        self._cur_u = 0.0
        self._u_limit = us[-1]

        self._brake = self._braked = False

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

    def _find_xref(self, state: Car, dt: float) -> npt.NDArray[np.floating[Any]]:
        "find the closest point in the reference trajectory, and interpolate the reference trajectory within a horizon"
        while True:
            self._cur_u = self._nearist_point(state)

            # interpolate the reference trajectory
            v = np.sign(scipy.interpolate.splev(self._cur_u, self._tck)[2]) * state.velocity
            length = max(MIN_HORIZON_DISTANCE, max(0, v) * dt * HORIZON_LENGTH)
            ref_u = np.linspace(self._cur_u, self._cur_u + length, HORIZON_LENGTH + 1)
            ref_u = np.clip(ref_u, a_min=None, a_max=self._u_limit)
            xref = np.array(scipy.interpolate.splev(ref_u, self._tck)).T

            # self._direction_changing_us[i - 1] <= self._cur_u < self._direction_changing_us[i]
            i = np.searchsorted(self._direction_changing_us, self._cur_u, side="right")
            changing_point = self._direction_changing_us[i] if i < len(self._direction_changing_us) else np.inf
            if ref_u[-1] >= changing_point:  # if the reference trajectory contains a direction change

                # if the direction change happens immediately after the first point, we discard the first point
                # and start to track the trajectory from the direction changing point
                if ref_u[1] >= changing_point:
                    self._cur_u = changing_point
                    continue

                # otherwise, we make the direction changing point to have zero velocity, and the vehicle should stop at that point
                i = np.searchsorted(ref_u, changing_point, side="right")
                xref = xref[:i]
                xref = np.vstack([xref, np.array(scipy.interpolate.splev(changing_point, self._tck)).T])
                xref = np.pad(xref, ((0, HORIZON_LENGTH + 1 - len(xref)), (0, 0)), mode="edge")
                xref[i:, 2] = xref[0, 2]
                xref[-1, 2] = state.velocity * -0.5

            if self._brake:
                if not self._braked:
                    brake_length = np.square(state.velocity) / (2 * Car.MAX_ACCEL * DESIRED_MAX_ACCEL_RATIO)
                    self._u_limit = min(self._u_limit, self._cur_u + brake_length, changing_point)
                    self._braked = True
                xref[:, 2] = 0.0
                xref[-1, 2] = state.velocity * -0.5
            elif ref_u[-1] == self._u_limit:
                # make the goal point to have zero velocity
                xref[ref_u == self._u_limit, 2] = 0
                xref[-1, 2] = state.velocity * -0.5

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
        return MPCResult(controls, states[:, [0, 1, 3, 2]], xref[:, [0, 1, 3, 2]])

    def brake(self) -> None:
        self._brake = True
