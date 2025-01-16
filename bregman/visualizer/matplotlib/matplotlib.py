import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase

from bregman.base import Coords, DualCoords, Point
from bregman.manifold.bisector import Bisector
from bregman.manifold.geodesic import BregmanGeodesic, Geodesic
from bregman.manifold.manifold import BregmanManifold
from bregman.visualizer.visualizer import (
    BregmanVisualizer,
    MultiBregmanVisualizer,
)


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
        resolution: Resolution of calculating curves in the visualizer.
        frames: Number of frames used in animation.
        intervals: Delay between frames in milliseconds.
        fig: Matplotlib figure for plotting.
        ax: Matplotlib axis for plotting.
        update_func_list: Update function list for matplotlib animation.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        plot_dims: tuple[int, int] = (0, 1),
        dim_names: tuple[str, str] | None = None,
        fig: FigureBase | None = None,
        ax: Axes | None = None,
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

        if fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        else:
            plt.style.use("bmh")
            self.fig, self.ax = plt.subplots()  # pyright: ignore

            self.fig: FigureBase
            self.ax: Axes

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
                xys = point.data[np.array([self.dim1, self.dim2])]
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

        rdelta = 1 / (self.resolution - 1)

        geodesic_points = [
            self.manifold.convert_coord(coords, geodesic(rdelta * t))
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

        x, y = bis_point.data[np.array([self.dim1, self.dim2])]
        data_lim = self.calculate_lims(bisector.coords, 0.2)

        y1 = (-w - x * data_lim.xmin) / y
        y2 = (-w - x * data_lim.xmax) / y

        p1_data = jnp.array([data_lim.xmin, y1])
        p2_data = jnp.array([data_lim.xmax, y2])

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
        delta = 1 / (self.frames - 1)
        geodesic_points = [
            self.manifold.convert_coord(coords, geodesic(delta * t))
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

    def _plot(self, coords: Coords) -> None:
        """Shared plotting script."""
        self.update_func_list = []

        # Static plots
        super().visualize(coords)

        # Setup figure
        self.ax.set_xlabel(self.dim_names[0])
        self.ax.set_ylabel(self.dim_names[1])

        if len(self.ax.get_legend_handles_labels()[0]) > 0:
            self.ax.legend()

    def _animate(self, coords: Coords) -> animation.FuncAnimation:
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

        return ani

    def visualize(self, coords: Coords) -> None:
        """Visualize all registered plots and animations and then run
        matplotlib's show function.

        Args:
            coords: Coordinate the visualization is being made in.
        """
        # Clear already visualized stuff on ax
        self.ax.clear()

        # Plot
        self._plot(coords)

        # Animation if specified
        if self.update_func_list:
            _ = self._animate(coords)

        plt.show()

    def save(self, coords: Coords, path: Path | str) -> None:
        """Visualize all registered plots and then save the visualization for
        the designated path.

        Args:
            coords: Coordinate the visualization is being made in.
            path: Save path for visualization.
        """
        # Clear already visualized stuff on ax
        self.ax.clear()

        # Plot
        self._plot(coords)

        if type(path) is str:
            path = Path(path)

        plt.tight_layout()
        plt.savefig(path)

    def save_gif(self, coords: Coords, path: Path | str) -> None:
        """Visualize all registered plots and then save the animation for
        the designated path.

        Args:
            coords: Coordinate the visualization is being made in.
            path: Save path for animation.
        """
        # Clear already visualized stuff on ax
        self.ax.clear()

        # Plot
        self._plot(coords)
        ani = self._animate(coords)

        if type(path) is str:
            path = Path(path)

        plt.tight_layout()
        ani.save(path, dpi=300, writer=animation.PillowWriter(fps=25))


class MultiMatplotlibVisualizer(MultiBregmanVisualizer[MatplotlibVisualizer]):
    def __init__(
        self,
        nrows: int,
        ncols: int,
        resolution: int = 120,
        frames: int = 120,
        intervals: int = 1,
        **kwargs,
    ):
        super().__init__(nrows, ncols)

        plt.style.use("bmh")
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
        self.fig = cast(plt.Figure, fig)

        self.axes: list[list[Axes]] = []
        for i in range(nrows):
            self.axes.append([])
            for j in range(ncols):
                self.axes[i].append(axes.flatten()[j * nrows + i])

        self.resolution = resolution
        self.frames = frames
        self.intervals = intervals

    def new_visualizer(
        self,
        row_idx: int,
        col_idx: int,
        manifold: BregmanManifold,
        coord: Coords,
        name: str = "",
        **kwargs,
    ) -> MatplotlibVisualizer:
        """Set new visualizer at position (row_idx, col_idx).

        Args:
            row_idx: Row index.
            col_idx: Col index.
            manifold: Bregman manifold in which geometric objects are being plotted in.
            coords: Coordinates the visualizer is plotting at.
            kwargs: Any other optional argument.
        """

        visualizer = MatplotlibVisualizer(
            manifold,
            fig=self.fig,
            ax=self.axes[row_idx][col_idx],
            resolution=self.resolution,
            frames=self.frames,
            intervals=self.intervals,
            **kwargs,
        )

        self.visualizations[row_idx][col_idx] = (coord, visualizer, name)
        return visualizer

    def copy_visualizer(
        self,
        from_row_idx: int,
        from_col_idx: int,
        to_row_idx: int,
        to_col_idx: int,
        coord: Coords | None = None,
        name: str | None = None,
    ) -> None:
        maybe_vis = self.visualizations[from_row_idx][from_col_idx]
        if maybe_vis is None:
            new_vis = None
        else:
            old_coord, vis, old_name = maybe_vis
            vis = copy.copy(vis)

            if coord is None:
                coord = old_coord

            if name is None:
                name = old_name

            vis.ax = self.axes[to_row_idx][to_col_idx]

            new_vis = (coord, vis, name)

        self.visualizations[to_row_idx][to_col_idx] = new_vis

    def visualize_all(self) -> None:
        """Visualize all visualizers in class."""

        combined_update_func_list = []
        for maybe_vis in itertools.chain.from_iterable(self.visualizations):
            if maybe_vis is None:
                continue

            coord, vis, name = maybe_vis

            vis._plot(coord)
            vis.ax.set_title(name)

            if vis.update_func_list:
                combined_update_func_list += vis.update_func_list

        # Animation if specified
        if combined_update_func_list:

            def update_all(frame: int):
                res = []
                for update in combined_update_func_list:
                    res.append(update(frame))

                return res

            ani = animation.FuncAnimation(
                fig=self.fig,
                func=update_all,
                frames=self.frames,
                interval=self.intervals,
            )

        plt.show()

    def save(self, path: Path | str) -> None:
        """Visualize all registered plots and then save the visualization for
        the designated path.

        Args:
            path: Save path for visualization.
        """
        for maybe_vis in itertools.chain.from_iterable(self.visualizations):
            if maybe_vis is None:
                continue

            coord, vis, name = maybe_vis

            vis._plot(coord)
            vis.ax.set_title(name)

        if type(path) is str:
            path = Path(path)

        plt.tight_layout()
        plt.savefig(path)
