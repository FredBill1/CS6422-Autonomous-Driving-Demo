from typing import Any

import numpy as np
import numpy.typing as npt


def wrap_angle(radian: npt.ArrayLike, zero_to_2pi: bool = False) -> npt.ArrayLike | npt.NDArray[np.floating[Any]]:
    ret = np.remainder(radian, 2 * np.pi)
    if zero_to_2pi:
        return ret
    ret = np.where(ret >= np.pi, ret - 2 * np.pi, ret)
    return ret if ret.shape else ret[()]


def smooth_yaw(yaws: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    diff = np.diff(yaws, prepend=0)
    diff = wrap_angle(diff)
    return np.cumsum(diff)
