from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from modeling.Car import Car
from modeling.Obstacles import Obstacles

NUM_PARTICLES = 500
NUM_PARTICLE_THRESHOLD = 250


def gaussian(x: npt.NDArray[np.floating[Any]], sigma: float) -> npt.NDArray[np.floating[Any]]:
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def calc_weights(
    particles: npt.NDArray[np.floating[Any]], obstacles: Obstacles, true_scan: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    assert true_scan.ndim == 2 and true_scan.shape[1] == 2, "True scan must be a 2D array of shape (n, 2)"
    SCAN_SIZE = true_scan.shape[0]
    if SCAN_SIZE == 0:
        return np.full(particles.shape[0], 1.0 / particles.shape[0])
    scans = np.empty((particles.shape[0], SCAN_SIZE, 2))
    for i, (xy, cy, sy) in enumerate(zip(particles[:, :2], np.cos(particles[:, 2]), np.sin(particles[:, 2]))):
        scans[i] = true_scan @ np.array([[cy, -sy], [sy, cy]]) + xy
    scans = scans.reshape(-1, 2)
    dists, _ = obstacles.kd_tree.query(scans, k=1)
    weights = np.exp(-np.square(dists / Car.LIDAR_SIGMA))
    weights = np.sum(weights.reshape(-1, SCAN_SIZE) ** 3, axis=1) + 1
    return weights


class PFResult(NamedTuple):
    state: Car
    particles: npt.NDArray[np.floating[Any]]
    weights: npt.NDArray[np.floating[Any]]


class ParticleFilter:
    def __init__(self, initial_state: npt.ArrayLike) -> None:
        assert np.shape(initial_state) == (3,), "Initial state must be an array containing [x, y, yaw]"
        self.particles = np.array([initial_state] * NUM_PARTICLES)
        self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)

    def update(
        self, control: tuple[float, float], obstacles: Obstacles, true_scan: npt.NDArray[np.floating[Any]], dt: float
    ) -> PFResult:
        particles = self.particles
        Car(*particles.T, *control).update(dt, with_noise=True)
        self.weights *= calc_weights(particles, obstacles, true_scan)
        self.weights /= np.sum(self.weights)
        estimated_state = particles.T @ self.weights

        if 1.0 / (self.weights @ self.weights) < NUM_PARTICLE_THRESHOLD:
            indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=self.weights)
            self.particles = particles[indices]
            self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        return PFResult(Car(*estimated_state, *control), particles, self.weights)
