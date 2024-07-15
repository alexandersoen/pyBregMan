import numpy as np

from bregman.application.distribution.exponential_family.gaussian.gaussian import \
    GaussianManifold
from bregman.base import (LAMBDA_COORDS, BregmanObject, Coords, DualCoords,
                          Point)
from bregman.visualizer.matplotlib.matplotlib import MatplotlibVisualizer
from bregman.visualizer.visualizer import VisualizerCallback


class VisualizeGaussian2DCovariancePoints(
    VisualizerCallback[MatplotlibVisualizer]
):
    """Callback to visualize the covariance matrix of points in 2D Gaussian
    manifolds.

    Parameters:
        scale: Scale of the radius of the covariance ellipsoid being plotted.
        npoints: Number of points used to generate the covariance ellipsoid.
    """

    def __init__(self, scale: float = 0.2, npoints: int = 1_000) -> None:
        """Initialize callback for covariance matrix visualizer.

        Args:
            scale: Scale of the radius of the covariance ellipsoid being plotted.
            npoints: Number of points used to generate the covariance ellipsoid.
        """
        super().__init__()

        self.scale = scale
        self.npoints = npoints

    def call(
        self,
        obj: BregmanObject,
        coords: Coords,
        visualizer: MatplotlibVisualizer,
        **kwargs,
    ) -> None:
        r"""Plots ellipsoid of the covariance matrix for points in 2D Gaussian
        manifolds. The callback accepts any type of BregmanObject and only
        creates this visualization if it is a Point of compatible dimension.

        Visualization only occurs in the :math:`\lambda`-coordinates.

        Args:
            obj: Objects being plotted.
            coords: Coordinates the covariance matrix is being evaluated on.
            visualizer: Visualizer the callback is being called in.
            **kwargs: Additional kwargs passed to matplotlib plot function.

        Raises:
            ValueError: Raises exception when visualizer's manifold is incompatible.
        """
        if type(visualizer.manifold) is not GaussianManifold:
            raise ValueError(
                f"Visualizer {visualizer}'s manifold type is not compatible with {self}"
            )

        if visualizer.manifold.input_dimension != 2:
            raise ValueError(
                f"Input dimension {visualizer.manifold.input_dimension} != 2"
            )

        if coords != LAMBDA_COORDS or not isinstance(obj, Point):
            return

        # dim = int(0.5 * (np.sqrt(4 * visualizer.manifold.dimension + 1) - 1))

        d_point = visualizer.manifold.convert_to_display(obj)
        mu = d_point.mu
        Sigma = d_point.Sigma

        L = np.linalg.cholesky(Sigma).T

        p = np.arange(self.npoints + 1)
        thetas = 2 * np.pi * p / self.npoints

        v = self.scale * np.column_stack([np.cos(thetas), np.sin(thetas)]).dot(
            L.T
        )
        v = v + mu

        tissot_kwargs = {
            "c": kwargs["c"] if "c" in kwargs else None,
            "alpha": kwargs["alpha"] * 0.5 if "alpha" in kwargs else 0.5,
            "ls": kwargs["ls"] if "ls" in kwargs else "--",
            "zorder": 0,
        }
        visualizer.ax.plot(v[:, 0], v[:, 1], **tissot_kwargs)


class Visualize2DTissotIndicatrix(VisualizerCallback[MatplotlibVisualizer]):
    """Callback to visualize the Tissot indicatrix for the metric tensor of 2D
    manifolds.

    Parameters:
        scale: Scale of the radius of the Tissot indicatrix ellipsoid being plotted.
        npoints: Number of points used to generate the Tissot indicatrix.
    """

    def __init__(self, scale: float = 0.1, npoints: int = 1_000) -> None:
        """Initialize callback for Tissot indicatrix visualizer.

        Args:
            scale: Scale of the radius of the Tissot indicatrix ellipsoid being plotted.
            npoints: Number of points used to generate the Tissot indicatrix.
        """
        super().__init__()

        self.scale = scale
        self.npoints = npoints

    def call(
        self,
        obj: BregmanObject,
        coords: Coords,
        visualizer: MatplotlibVisualizer,
        **kwargs,
    ) -> None:
        r"""Plots ellipsoid of the Tissot indicatrix for points in 2D Bregman
        manifolds. The callback accepts any type of BregmanObject and only
        creates this visualization if it is a Point of compatible dimension.

        Visualization only occurs in the :math:`\lambda`-coordinates.

        Args:
            obj: Objects being plotted.
            coords: Coordinates the Tissot indicatrix is being evaluated on.
            visualizer: Visualizer the callback is being called in.
            **kwargs: Additional kwargs passed to matplotlib plot function.

        Raises:
            ValueError: Raises exception when visualizer's manifold is incompatible.
        """

        if visualizer.manifold.dimension != 2:
            raise ValueError(f"Dimension {visualizer.manifold.dimension} != 2")

        if coords == LAMBDA_COORDS or not isinstance(obj, Point):
            return None

        point = visualizer.manifold.convert_coord(coords, obj)

        metric = visualizer.manifold.bregman_connection(
            DualCoords(coords)
        ).metric(point.data)

        L = np.linalg.cholesky(metric).T

        p = np.arange(self.npoints + 1)
        thetas = 2 * np.pi * p / self.npoints

        v = self.scale * np.column_stack([np.cos(thetas), np.sin(thetas)]).dot(
            L.T
        )
        v = v + point.data

        tissot_kwargs = {
            "c": kwargs["c"] if "c" in kwargs else None,
            "alpha": kwargs["alpha"] if "alpha" in kwargs else None,
            "ls": kwargs["ls"] if "ls" in kwargs else "--",
            "zorder": 0,
        }
        visualizer.ax.plot(v[:, 0], v[:, 1], **tissot_kwargs)
