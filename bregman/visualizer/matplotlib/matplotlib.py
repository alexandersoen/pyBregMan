from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bregman.base import Coords, DualCoords, Point
from bregman.manifold.bisector import Bisector
from bregman.manifold.geodesic import BregmanGeodesic, Geodesic
from bregman.manifold.manifold import BregmanManifold
from bregman.visualizer.visualizer import BregmanVisualizer


@dataclass
class DataLim:
    """The axis limits of the data being visualizer.

    Parameters:
        coords: Coordinate the data is being visualized in.
        xmin: Minimum x-values.
        xmax: Maximum x-values.
        ymin: Minimum y-values.
        ymax: Maximum y-values.
    """

    coords: Coords
    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray


class MatplotlibVisualizer(BregmanVisualizer):
    """Visualization class using matplotlib as a visualization back-end.
    Allows the visualization of geometric objects in a Bregman objects over two
    dimension axis.

    Parameters:
        dim1: Axis 1 for visualization.
        dim2: Axis 2 for visualization.
        dim_names: Tuple of names used to describe visualization axis.
        resolution: Resolution of calculating curves in the visualizer.
        frames: Number of frames used in animation.
        intervals: Delay between frames in milliseconds.
        rdelta: Resolution time delta for evaluating curves parameterized in [0, 1].
        delta: Animation time delta for evaluating curves parameterized in [0, 1].
        fig: Matplotlib figure for plotting.
        ax: Matplotlib axis for plotting.
        update_func_list: Update function list for matplotlib animation.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        plot_dims: tuple[int, int],
        dim_names: tuple[str, str] | None = None,
        resolution: int = 120,
        frames: int = 120,
        intervals: int = 1,
    ) -> None:
        """Initialize matplotlib visualization class.

        Args:
            manifold: Bregman manifold in which geometric objects are being plotted in.
            plot_dims: Dimensions which are being plotted by visualizer.
            dim_names: Axis labels / names of the dimensions being plotted.
            resolution: Resolution of calculating curves in the visualizer.
            frames: Number of frames used in animation.
            intervals: Delay between frames in milliseconds.
        """
        super().__init__(manifold)

        self.dim1 = plot_dims[0]
        self.dim2 = plot_dims[1]

        if dim_names is None:
            dim_names = (f"Dim {plot_dims[0]}", f"Dim {plot_dims[1]}")
        self.dim_names = dim_names

        self.resolution = resolution
        self.frames = frames
        self.intervals = intervals

        self.rdelta = 1 / (self.resolution - 1)
        self.delta = 1 / (self.frames - 1)

        self.fig: Figure
        self.ax: Axes

        plt.style.use("bmh")
        self.fig, self.ax = plt.subplots()

        self.update_func_list: list[Callable[[int], Any]] = []

    def calculate_lims(self, coords: Coords, cut: float = 1.0) -> DataLim:
        """Calculate the data limits (data axis bounds) for the current objects
        which will be plotted in the visualizer.

        Args:
            coords: Coordinate for object to be plotted in.
            cut: Cut percentage to remove from axis bounds being calculated.

        Returns:

        """
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

    def plot_point(self, coords: Coords, point: Point, **kwargs) -> None:
        """Matplotlib plotting function for Point objects.

        Args:
            coords: Coordinate for object to be plotted in.
            point: Point object to be plotted.
            **kwargs: Additional kwargs passed to matplotlib scatter function.
        """
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
        self, coords: Coords, geodesic: Geodesic, **kwargs
    ) -> None:
        """Matplotlib plotting function for Geodesic objects.

        Args:
            coords: Coordinate for object to be plotted in.
            geodesic: Geodesic object to be plotted.
            **kwargs: Additional kwargs passed to matplotlib plot function.
        """
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
        self, coords: Coords, bisector: Bisector, **kwargs
    ) -> None:
        """Matplotlib plotting function for Bisector objects.

        Args:
            coords: Coordinate for object to be plotted in.
            bisector: Bisector object to be plotted.
            **kwargs: Additional kwargs passed to matplotlib plot function.
        """

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
            dcoords=DualCoords(bisector.coords),
        )
        self.plot_geodesic(coords, plot_geo, **bis_vis_kwargs)

    def animate_geodesic(
        self, coords: Coords, geodesic: Geodesic, **kwargs
    ) -> None:
        """Matplotlib animation function for Geodesic objects.

        Args:
            coords: Coordinate for object to be animted in.
            geodesic: Bisector object to be animated.
            **kwargs: Additional kwargs passed to matplotlib plot function.
        """
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

    def visualize(self, coords: Coords) -> None:
        """Visualize all registered plots and animations and then run
        matplotlib's show function.

        Args:
            coords: Coordinate the visualization is being made in.
        """
        self.update_func_list = []

        # Static plots
        super().visualize(coords)

        # Animation if specified
        if self.update_func_list:

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

        # Setup figure
        self.ax.set_xlabel(self.dim_names[0])
        self.ax.set_ylabel(self.dim_names[1])

        if len(self.ax.get_legend_handles_labels()[0]) > 0:
            self.ax.legend()

        plt.show()

    def save(self, coords: Coords, path: Path | str) -> None:
        """Visualize all registered plots and then save the visualization for
        the designated path.

        Args:
            coords: Coordinate the visualization is being made in.
            path: Save path for visualization.
        """
        if path is str:
            path = Path(path)

        self.update_func_list = []
        super().visualize(coords)

        if len(self.ax.get_legend_handles_labels()[0]) > 0:
            self.ax.legend()

        plt.savefig(path)
