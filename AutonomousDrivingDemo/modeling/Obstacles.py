from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


class ObstacleGrid(NamedTuple):
    minx: float
    maxx: float
    miny: float
    maxy: float
    resolution: float
    grid: np.ndarray

    def calc_index(self, xy: npt.ArrayLike) -> tuple[int, int]:
        x, y = xy
        return int((y - self.miny) / self.resolution), int((x - self.minx) / self.resolution)


class Obstacles:
    def __init__(self, coordinates: npt.NDArray[np.floating[Any]]) -> None:
        assert coordinates.ndim == 2 and coordinates.shape[1] == 2, "Coordinates must be a 2D array of shape (n, 2)"
        self.coordinates = coordinates
        self.kd_tree = KDTree(coordinates)

    def downsampling_to_grid(self, resolution: float, radius: float) -> ObstacleGrid:
        half_res = resolution / 2
        minx, maxx = np.min(self.coordinates[:, 0]) - half_res, np.max(self.coordinates[:, 0]) + half_res
        miny, maxy = np.min(self.coordinates[:, 1]) - half_res, np.max(self.coordinates[:, 1]) + half_res
        x_count, y_count = round((maxx - minx) / resolution), round((maxy - miny) / resolution)
        maxx, maxy = minx + x_count * resolution, miny + y_count * resolution
        points = np.array(
            np.meshgrid(
                np.arange(minx + half_res, maxx, resolution),
                np.arange(miny + half_res, maxy, resolution),
                indexing="ij",
            )
        ).T.reshape(-1, 2)
        dist, _ = self.kd_tree.query(points, k=1, distance_upper_bound=radius + resolution)
        grid = (dist <= radius).reshape(y_count, x_count)
        return ObstacleGrid(minx, maxx, miny, maxy, resolution, grid)
