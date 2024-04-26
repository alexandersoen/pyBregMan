from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bregman.base import Coordinates, Point
from bregman.manifold.manifold import BregmanManifold, Geodesic
from bregman.visualizer.visualizer import Visualizer


class MatplotlibVisualizer(Visualizer):

    def __init__(
        self,
        manifold: BregmanManifold,
        plot_dims: tuple[int, int],
        resolution: int = 120,
        frames: int = 120,
        intervals: int = 1,
    ) -> None:
        super().__init__(manifold)

        self.dim1 = plot_dims[0]
        self.dim2 = plot_dims[1]

        self.resolution = resolution
        self.frames = frames
        self.intervals = intervals

        self.delta = 1 / (self.frames - 1)

        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots()

        self.update_func_list: list[Callable[[int], Any]] = []

    def plot_point(self, coords: Coordinates, point: Point, **kwargs) -> None:
        point = self.manifold.convert_coord(coords, point)
        self.ax.scatter(point.data[self.dim1], point.data[self.dim2], **kwargs)

    def plot_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        geo_vis_kwargs = {
            "ls": "--",
            "linewidth": 1,
        }
        geo_vis_kwargs.update(kwargs)

        geodesic_points = [
            self.manifold.convert_coord(coords, geodesic(self.delta * t))
            for t in range(self.resolution)
        ]
        geodesic_data = np.vstack([p.data for p in geodesic_points])

        self.ax.plot(
            geodesic_data[:, self.dim1],
            geodesic_data[:, self.dim2],
            **geo_vis_kwargs
        )

    def animate_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        geodesic_points = [
            self.manifold.convert_coord(coords, geodesic(self.delta * t))
            for t in range(self.frames)
        ]
        geodesic_data = np.vstack([p.data for p in geodesic_points])

        geodesic_pt = self.ax.scatter(
            geodesic_data[0, self.dim1], geodesic_data[0, self.dim2], **kwargs
        )

        def update(frame: int):
            geodesic_pt.set_offsets(
                geodesic_data[frame, [self.dim1, self.dim2]]
            )
            return geodesic_pt

        self.update_func_list.append(update)

    def visualize(self, coords: Coordinates) -> None:
        self.update_func_list = []
        super().visualize(coords)

        def update_all(frame: int):
            res = []
            for update in self.update_func_list:
                res.append(update(frame))

            return res

        ani = animation.FuncAnimation(
            fig=self.fig,
            func=update_all,
            frames=self.frames,
            interval=self.intervals,
        )

        self.ax.legend()
        plt.show()
