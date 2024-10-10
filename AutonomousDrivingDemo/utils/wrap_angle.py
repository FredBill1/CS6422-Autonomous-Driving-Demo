from typing import Any

import numpy as np
import numpy.typing as npt


def wrap_angle(radian: npt.ArrayLike, zero_to_2pi: bool = False) -> npt.ArrayLike | npt.NDArray[np.floating[Any]]:
    """
    Wrap the angle of any range to [-pi, pi] or [0, 2pi].

    E.g.
    >>> assert wrap_angle(2.5 * np.pi) == 0.5 * np.pi
    >>> assert wrap_angle(-0.5 * np.pi, zero_to_2pi=True) == 1.5 * np.pi
    """
    ret = np.remainder(radian, 2 * np.pi)
    if zero_to_2pi:
        return ret
    ret = np.where(ret >= np.pi, ret - 2 * np.pi, ret)
    return ret if ret.shape else ret[()]


def smooth_yaw(yaws: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    "Make the yaws along a trajectory continuous, preventing sudden changes of -2pi -> 2pi"
    diff = np.diff(yaws, prepend=0)
    diff = wrap_angle(diff)
    return np.cumsum(diff)
