from typing import Any

import numpy as np
import numpy.typing as npt


def estimate_rigid_transform(
    A: npt.NDArray[np.floating[Any]], B: npt.NDArray[np.floating[Any]]
) -> tuple[: npt.NDArray[np.floating[Any]], : npt.NDArray[np.floating[Any]]]:
    "Sorkine-Hornung O, Rabinovich M. Least-squares rigid motion using svd[J]. Computing, 2017, 1(1): 1-5."
    assert len(A.shape) == 2 and A.shape[1] == 2, f"Expected 2D array with shape (n, 2), got {A.shape=}"
    assert A.shape == B.shape, f"Expected {A.shape=} to be equal to {B.shape=}"
    c_A, c_B = np.mean(A, axis=0), np.mean(B, axis=0)  # Compute the centroids
    A, B = A - c_A, B - c_B  # Compute the centered vectors
    H = A.T @ B  # Compute the covariance matrix
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T  # Rotation matrix
    if np.linalg.det(R) < 0:  # Ensure a right-handed coordinate system (det(R) should be 1)
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_B - R @ c_A  # translation vector
    return R, t
