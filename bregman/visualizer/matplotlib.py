from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bregman.base import LAMBDA_COORDS, Coordinates, CoordObject, Point
from bregman.manifold.bisector import Bisector
from bregman.manifold.distribution.exponential_family.gaussian import \
    _flatten_to_mu_Sigma
from bregman.manifold.manifold import BregmanManifold, DualCoord, Geodesic
from bregman.manifold.parallel_transport import ParallelTansport
from bregman.visualizer.visualizer import (CoordObjectVisualizer,
                                           VisualizerCallback)


class CoordObjectMatplotlibVisualizer(CoordObjectVisualizer):

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

        self.geodesic_rate = 0.1

        self.resolution = resolution
        self.frames = frames
        self.intervals = intervals

        self.delta = 1 / (self.frames - 1)

        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots()

        self.update_func_list: list[Callable[[int], Any]] = []

        # Edges of visualizer
        self.xmin, self.xmax = (0, 0)
        self.ymin, self.ymax = (0, 0)

    def calculate_lims(self, coords: Coordinates) -> None:
        xys_list = []

        for obj, _ in self.plot_list:
            if isinstance(obj, Point):

                point = self.manifold.convert_coord(coords, obj)
                xys = point.data[[self.dim1, self.dim2]]
                xys_list.append(xys)

        xys_data = np.vstack(xys_list)

        self.xmin, self.ymin = np.min(xys_data, axis=0)
        self.xmax, self.ymax = np.max(xys_data, axis=0)
        print(self.xmin, self.ymin, self.xmax, self.ymax)

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
            **geo_vis_kwargs,
        )

    def plot_bisector(
        self, coords: Coordinates, bisector: Bisector, **kwargs
    ) -> None:
        if bisector.coords != coords:
            return None

        bis_vis_kwargs = {
            "ls": "-.",
            "linewidth": 1,
        }
        bis_vis_kwargs.update(kwargs)

        bis_point = bisector.bisect_proj_point()
        w = bisector.w()

        print(bis_point)
        print(w)

        x, y = bis_point.data[[self.dim1, self.dim2]]

        y1 = (w - x * self.xmin) / y
        y2 = (w - x * self.xmax) / y

        self.ax.plot([self.xmin, self.xmax], [y1, y2], **bis_vis_kwargs)

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

            #             if frame % int(self.frames * self.geodesic_rate) == 0:
            #                 cur_geo_point = geodesic_points[frame]
            #                 cur_kwargs = {
            #                     "c": kwargs["c"] if "c" in kwargs else None,
            #                     "alpha": frame / self.frames,
            #                 }
            #                 self.plot_point(coords, cur_geo_point, **cur_kwargs)
            #                 self.run_callbacks(cur_geo_point, coords, **cur_kwargs)

            return geodesic_pt

        self.update_func_list.append(update)

    def animate_parallel_transport(
        self,
        coords: Coordinates,
        parallel_transport: ParallelTansport,
        **kwargs,
    ) -> None:
        pt_vectors = [
            parallel_transport(self.delta * t) for t in range(self.frames)
        ]
        p1_data = np.vstack(
            [
                self.manifold.convert_coord(coords, v[0]).data
                for v in pt_vectors
            ]
        )
        p2_data = np.vstack(
            [
                self.manifold.convert_coord(coords, v[1]).data
                for v in pt_vectors
            ]
        )

        (pt_line,) = self.ax.plot(
            [p1_data[0, self.dim1], p2_data[0, self.dim1]],
            [p1_data[0, self.dim2], p2_data[0, self.dim2]],
            **kwargs,
        )

        def update(frame: int):
            pt_line.set_data(
                [p1_data[frame, self.dim1], p2_data[frame, self.dim1]],
                [p1_data[frame, self.dim2], p2_data[frame, self.dim2]],
            )
            return pt_line

        self.update_func_list.append(update)

    def visualize(self, coords: Coordinates) -> None:
        self.update_func_list = []
        self.calculate_lims(coords)
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


class VisualizeGaussian2DCovariancePoints(
    VisualizerCallback[CoordObjectMatplotlibVisualizer]
):

    def __init__(self, scale: float = 0.2, npoints: int = 1_000) -> None:
        super().__init__()

        self.scale = scale
        self.npoints = npoints

    def call(
        self,
        obj: CoordObject,
        coords: Coordinates,
        visualizer: CoordObjectMatplotlibVisualizer,
        **kwargs,
    ) -> None:

        if coords != LAMBDA_COORDS or not isinstance(obj, Point):
            return None

        dim = int(0.5 * (np.sqrt(4 * visualizer.manifold.dimension + 1) - 1))

        mu, Sigma = _flatten_to_mu_Sigma(
            dim, visualizer.manifold.convert_coord(LAMBDA_COORDS, obj).data
        )

        L = np.linalg.cholesky(Sigma).T

        p = np.arange(self.npoints + 1)
        thetas = 2 * np.pi * p / self.npoints

        v = self.scale * np.column_stack([np.cos(thetas), np.sin(thetas)]).dot(
            L.T
        )
        v = v + mu

        visualizer.ax.plot(v[:, 0], v[:, 1], **kwargs)


class Visualize2DTissotIndicatrix(
    VisualizerCallback[CoordObjectMatplotlibVisualizer]
):

    def __init__(self, scale: float = 0.2, npoints: int = 1_000) -> None:
        super().__init__()

        self.scale = scale
        self.npoints = npoints

    def call(
        self,
        obj: CoordObject,
        coords: Coordinates,
        visualizer: CoordObjectMatplotlibVisualizer,
        **kwargs,
    ) -> None:

        if coords == LAMBDA_COORDS or not isinstance(obj, Point):
            return None

        point = visualizer.manifold.convert_coord(coords, obj)

        metric = visualizer.manifold.bregman_connection(
            DualCoord(coords)
        ).metric(point.data)

        L = np.linalg.cholesky(metric).T

        p = np.arange(self.npoints + 1)
        thetas = 2 * np.pi * p / self.npoints

        v = self.scale * np.column_stack([np.cos(thetas), np.sin(thetas)]).dot(
            L.T
        )
        v = v + point.data

        visualizer.ax.plot(v[:, 0], v[:, 1], **kwargs)
