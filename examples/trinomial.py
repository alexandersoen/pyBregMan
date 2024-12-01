"""
Trinomial Visualization
"""

from functools import partial
from jax import Array
import jax.numpy as jnp

from bregman.application.distribution.exponential_family.categorical import (
    CategoricalManifold,
)
from bregman.application.distribution.exponential_family.multinomial.multinomial import (
    MultinomialManifold,
)
from bregman.base import (
    ETA_COORDS,
    LAMBDA_COORDS,
    THETA_COORDS,
    Coords,
    DualCoords,
    Point,
)
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.visualizer.matplotlib.matplotlib import MatplotlibVisualizer


EQT_COORDS = Coords("eqt", latex_name=r"\triangle")


def add_equilateral_triangle_coords(
    manifold: MultinomialManifold,
) -> MultinomialManifold:
    if manifold.k != 3:
        raise ValueError(
            "Can only add equilateral triangle coordinates for 3-dimension MultinomialManifold"
        )

    def _eta_to_eqt(eta: Array) -> Array:
        t1 = jnp.array((1.0, 0.5))
        return jnp.array([jnp.dot(eta, t1), eta[1] * jnp.sqrt(3.0) / 2.0])

    def _lambda_to_eqt(lamb: Array) -> Array:
        eta = manifold._lambda_to_eta(lamb)
        return _eta_to_eqt(eta)

    def _theta_to_eqt(theta: Array) -> Array:
        eta = manifold._theta_to_eta(theta)
        return _eta_to_eqt(eta)

    manifold.atlas.add_coords(EQT_COORDS)

    manifold.atlas.add_transition(ETA_COORDS, EQT_COORDS, _eta_to_eqt)
    manifold.atlas.add_transition(THETA_COORDS, EQT_COORDS, _theta_to_eqt)
    manifold.atlas.add_transition(LAMBDA_COORDS, EQT_COORDS, _lambda_to_eqt)

    return manifold


def equilateral_triangle_boundary_preprocess(visualizer: MatplotlibVisualizer):
    if (
        not isinstance(visualizer.manifold, MultinomialManifold)
        or visualizer.manifold.k != 3
    ):
        raise ValueError(
            "Can only add equilateral triangle boundary for 3-dimension MultinomialManifold"
        )

    scale = manifold.n

    b_xs = scale * [0.0, 1.5, 0.75, 0.0]
    b_ys = scale * [0.0, 0.0, jnp.sqrt(3.0) / 2.0, 0.0]

    visualizer.ax.plot(b_xs, b_ys)

    visualizer.ax.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
    )


if __name__ == "__main__":

    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(3)
    manifold = add_equilateral_triangle_coords(manifold)

    coord1 = Point(LAMBDA_COORDS, jnp.array([0.2, 0.4, 0.4]))
    coord2 = Point(LAMBDA_COORDS, jnp.array([0.5, 0.2, 0.3]))

    primal_geo = BregmanGeodesic(manifold, coord1, coord2, dcoords=DualCoords.THETA)
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, dcoords=DualCoords.ETA)

    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    visualizer.add_preprocess(equilateral_triangle_boundary_preprocess)

    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="red", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="blue", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="red")
    visualizer.animate_object(dual_geo, c="blue")

    visualizer.visualize(EQT_COORDS)
