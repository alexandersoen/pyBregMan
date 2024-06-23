from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bregman.base import Coordinates, Point
from bregman.manifold.bisector import Bisector
from bregman.manifold.geodesic import BregmanGeodesic, Geodesic
from bregman.manifold.manifold import BregmanManifold, DualCoord
from bregman.visualizer.visualizer import BregmanObjectVisualizer


@dataclass
class DataLim:
    coords: Coordinates
    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray


class BregmanObjectMatplotlibVisualizer(BregmanObjectVisualizer):

    def __init__(
        self,
        manifold: BregmanManifold,
        plot_dims: tuple[int, int],
        dim_names: tuple[str, str] = ("", ""),
        resolution: int = 120,
        frames: int = 120,
        intervals: int = 1,
    ) -> None:
        super().__init__(manifold)

        self.dim1 = plot_dims[0]
        self.dim2 = plot_dims[1]

        self.dim_names = dim_names

        self.geodesic_rate = 0.1

        self.resolution = resolution
        self.frames = frames
        self.intervals = intervals

        self.rdelta = 1 / (self.resolution - 1)
        self.delta = 1 / (self.frames - 1)

        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots()

        self.update_func_list: list[Callable[[int], Any]] = []

    def calculate_lims(self, coords: Coordinates, cut: float = 1.0) -> DataLim:
        xys_list = []

        for obj, _ in self.plot_list:
            if isinstance(obj, Point):

                point = self.manifold.convert_coord(coords, obj)
                xys = point.data[[self.dim1, self.dim2]]
                xys_list.append(xys)

        xys_data = np.vstack(xys_list)

        xmin, ymin = np.min(xys_data, axis=0)
        xmax, ymax = np.max(xys_data, axis=0)

        xcut = (xmax - xmin) * (1 - cut) / 2
        xmin = xmin + xcut
        xmax = xmax - xcut

        ycut = (ymax - ymin) * (1 - cut) / 2
        ymin = ymin + ycut
        ymax = ymax - ycut

        return DataLim(coords, xmin, xmax, ymin, ymax)

    def plot_point(self, coords: Coordinates, point: Point, **kwargs) -> None:
        kwargs = kwargs.copy()

        edgecolors = None
        facecolors = None
        if "marker" not in kwargs and "edgecolors" in kwargs:
            edgecolors = kwargs["edgecolors"]
            facecolors = "none"
        elif "marker" not in kwargs and "c" in kwargs:
            edgecolors = kwargs["c"]
            facecolors = "none"
            del kwargs["c"]

        point_vis_kwargs: dict[str, Any] = {
            "marker": "o",
            "edgecolors": edgecolors,
            "facecolors": facecolors,
        }
        point_vis_kwargs.update(kwargs)

        point = self.manifold.convert_coord(coords, point)
        self.ax.scatter(
            point.data[self.dim1], point.data[self.dim2], **point_vis_kwargs
        )

    def plot_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        geo_vis_kwargs: dict[str, Any] = {
            "ls": "-",
            "linewidth": 1,
        }
        geo_vis_kwargs.update(kwargs)

        geodesic_points = [
            self.manifold.convert_coord(coords, geodesic(self.rdelta * t))
            for t in range(self.resolution)
        ]
        geodesic_data = np.vstack([p.data for p in geodesic_points])

        self.ax.plot(
            geodesic_data[:, self.dim1],
            geodesic_data[:, self.dim2],
            **geo_vis_kwargs,
        )

    def plot_bisector(
        self, coords: Coordinates, bisector: Bisector, **kwargs
    ) -> None:

        bis_vis_kwargs: dict[str, Any] = {
            "ls": "-.",
            "linewidth": 1,
        }
        bis_vis_kwargs.update(kwargs)

        bis_point = bisector.bisect_proj_point()
        w = bisector.shift()

        x, y = bis_point.data[[self.dim1, self.dim2]]
        data_lim = self.calculate_lims(bisector.coords, 0.2)

        y1 = (-w - x * data_lim.xmin) / y
        y2 = (-w - x * data_lim.xmax) / y

        p1_data = np.array([data_lim.xmin, y1])
        p2_data = np.array([data_lim.xmax, y2])

        plot_geo = BregmanGeodesic(
            self.manifold,
            Point(bisector.coords, p1_data),
            Point(bisector.coords, p2_data),
            coord=DualCoord(bisector.coords),
        )
        self.plot_geodesic(coords, plot_geo, **bis_vis_kwargs)

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

        with plt.style.context("bmh"):
            ani = animation.FuncAnimation(
                fig=self.fig,
                func=update_all,
                frames=self.frames,
                interval=self.intervals,
            )

            self.ax.set_xlabel(self.dim_names[0])
            self.ax.set_ylabel(self.dim_names[1])

            self.ax.legend()
            plt.show()
