import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bregman.application.distribution.exponential_family.gaussian.gaussian import (
    GaussianManifold,
    GaussianPoint,
)
from bregman.base import (
    ETA_COORDS,
    LAMBDA_COORDS,
    THETA_COORDS,
    BregmanObject,
    Coords,
    DualCoords,
    Point,
)
from bregman.visualizer.matplotlib.matplotlib import MatplotlibVisualizer
from bregman.visualizer.visualizer import VisualizerCallback


def draw_ellipse(
    visualizer: MatplotlibVisualizer,
    mu: Array,
    cov: Array,
    npoints: int,
    scale: float,
    **kwargs,
) -> None:
    _cov = scale * cov
    L = np.linalg.cholesky(np.array(_cov)).T

    p = np.arange(npoints + 1)
    thetas = 2 * np.pi * p / npoints

    v = np.column_stack([np.cos(thetas), np.sin(thetas)]).dot(L.T)
    v = v + mu

    visualizer.ax.plot(v[:, 0], v[:, 1], **kwargs)


class VisualizeGaussian2DCovariancePoints(VisualizerCallback[MatplotlibVisualizer]):
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

        # Define point to plot
        d_point: GaussianPoint = visualizer.manifold.convert_to_display(obj)
        gaussian_kwargs = {
            "c": kwargs["c"] if "c" in kwargs else None,
            "alpha": kwargs["alpha"] * 0.5 if "alpha" in kwargs else 0.5,
            "ls": kwargs["ls"] if "ls" in kwargs else "--",
            "zorder": 0,
        }

        draw_ellipse(
            visualizer,
            d_point.mu,
            d_point.Sigma,
            self.npoints,
            self.scale,
            **gaussian_kwargs,
        )


class Visualize2DTissotIndicatrix(VisualizerCallback[MatplotlibVisualizer]):
    """Callback to visualize the Tissot indicatrix for the metric tensor of 2D
    manifolds.

    Parameters:
        scale: Scale of the radius of the Tissot indicatrix ellipsoid being plotted.
        npoints: Number of points used to generate the Tissot indicatrix.
    """

    def __init__(
        self, scale: float = 1.0, npoints: int = 1_000, inverse: bool = False
    ) -> None:
        """Initialize callback for Tissot indicatrix visualizer.

        Args:
            scale: Scale of the radius of the Tissot indicatrix ellipsoid being plotted.
            npoints: Number of points used to generate the Tissot indicatrix.
        """
        super().__init__()

        self.scale = scale
        self.npoints = npoints
        self.inverse = inverse

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

        if not isinstance(obj, Point):
            return None

        # For non-dual coordinates, we use the change of measure
        # calculation on the metric tensor.
        if coords != THETA_COORDS or coords != ETA_COORDS:
            tmp_coord = THETA_COORDS
            _point = visualizer.manifold.convert_coord(tmp_coord, obj)
            point = visualizer.manifold.convert_coord(coords, obj)

            _metric = visualizer.manifold.bregman_connection(
                DualCoords(tmp_coord)
            ).metric(_point.data)

            # Calculate change of measure Jacobian
            change_of_coords = visualizer.manifold.atlas.transitions[
                (coords, tmp_coord)
            ]
            change_jac = jax.jacobian(change_of_coords)(point.data)
            metric = jnp.einsum("ai,bj,ab->ij", change_jac, change_jac, _metric)

        else:
            point = visualizer.manifold.convert_coord(coords, obj)

            metric = visualizer.manifold.bregman_connection(DualCoords(coords)).metric(
                point.data
            )

        if self.inverse:
            metric = jnp.linalg.inv(metric)

        tissot_kwargs = {
            "c": kwargs["c"] if "c" in kwargs else None,
            "alpha": kwargs["alpha"] if "alpha" in kwargs else None,
            "ls": kwargs["ls"] if "ls" in kwargs else "--",
            "zorder": 0,
        }

        draw_ellipse(
            visualizer,
            point.data,
            metric,
            self.npoints,
            self.scale,
            **tissot_kwargs,
        )


class EllipseAtPoint(VisualizerCallback[MatplotlibVisualizer]):
    """Callback to visualize ellipse at points with fixed covariance.

    Parameters:
        cov: Covariance matrix being used.
        scale: Scale of the radius of the Tissot indicatrix ellipsoid being plotted.
        npoints: Number of points used to generate the Tissot indicatrix.
    """

    def __init__(
        self,
        cov: Array,
        cov_coord: Coords,
        scale: float = 1.0,
        npoints: int = 1_000,
    ) -> None:
        """Initialize callback for Tissot indicatrix visualizer.

        Args:
            cov: Covariance matrix being used.
            scale: Scale of the radius of the Tissot indicatrix ellipsoid being plotted.
            npoints: Number of points used to generate the Tissot indicatrix.
        """
        super().__init__()

        self.cov = cov
        self.cov_coord = cov_coord
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

        if not isinstance(obj, Point):
            return None

        # Calculate change of measure Jacobian
        point = visualizer.manifold.convert_coord(coords, obj)
        if self.cov_coord != coords:
            change_of_coords = visualizer.manifold.atlas.transitions[
                (coords, self.cov_coord)
            ]
            change_jac = jax.jacobian(change_of_coords)(point.data)
            cov = jnp.einsum("ai,bj,ab->ij", change_jac, change_jac, self.cov)
        else:
            cov = self.cov

        cov_kwargs = {
            "c": kwargs["c"] if "c" in kwargs else None,
            "alpha": kwargs["alpha"] if "alpha" in kwargs else None,
            "ls": kwargs["ls"] if "ls" in kwargs else "--",
            "zorder": 0,
        }

        draw_ellipse(
            visualizer,
            point.data,
            cov,
            self.npoints,
            self.scale,
            **cov_kwargs,
        )
