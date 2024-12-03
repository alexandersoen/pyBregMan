"""
Trinomial Visualization
"""

import jax.numpy as jnp
from jax import Array

from bregman.application.distribution.exponential_family.categorical import (
    CategoricalManifold,
)
from bregman.application.distribution.exponential_family.multinomial.geodesic import (
    FisherRaoMultinomialGeodesic,
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
from bregman.dissimilarity.bregman import ChernoffInformation
from bregman.manifold.bisector import BregmanBisector
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

    scale = visualizer.manifold.n

    b_xs = scale * [0.0, 1.0, 0.5, 0.0]
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

    visualizer.dim_names = ("", "")


def main() -> None:
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(3)
    manifold = add_equilateral_triangle_coords(manifold)

    coord1 = Point(LAMBDA_COORDS, jnp.array([0.2, 0.1, 0.7]))
    coord2 = Point(LAMBDA_COORDS, jnp.array([0.6, 0.2, 0.2]))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.THETA
    )
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, dcoords=DualCoords.ETA)
    fr_geo = FisherRaoMultinomialGeodesic(manifold, coord1, coord2)

    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    visualizer.add_preprocess(equilateral_triangle_boundary_preprocess)

    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="red", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="blue", label="Dual Geodesic")
    visualizer.plot_object(fr_geo, c="purple", label="Fisher-Rao Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="red")
    visualizer.animate_object(dual_geo, c="blue")
    visualizer.animate_object(fr_geo, c="purple")

    visualizer.save_gif(EQT_COORDS, "results/trinomial_fr.gif")
    visualizer.save(EQT_COORDS, "results/trinomial_fr.png")
    visualizer.visualize(EQT_COORDS)


def main_chernoff() -> None:
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(3)
    manifold = add_equilateral_triangle_coords(manifold)

    coord1 = Point(LAMBDA_COORDS, jnp.array([0.2, 0.1, 0.7]))
    coord2 = Point(LAMBDA_COORDS, jnp.array([0.6, 0.2, 0.2]))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.THETA
    )
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, dcoords=DualCoords.ETA)

    chernoff_point_alpha = ChernoffInformation(manifold).chernoff_point(
        coord1, coord2
    )
    chernoff_point = primal_geo(1 - chernoff_point_alpha)

    eta_bisector = BregmanBisector(
        manifold,
        coord1,
        coord2,
        dcoords=DualCoords.ETA,
    )

    theta_bisector = BregmanBisector(
        manifold,
        coord1,
        coord2,
        dcoords=DualCoords.THETA,
    )

    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    visualizer.add_preprocess(equilateral_triangle_boundary_preprocess)

    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(
        chernoff_point,
        label=f"Chernoff Point, alpha={chernoff_point_alpha:.2f}",
        marker="X",
    )
    visualizer.plot_object(primal_geo, c="red", label="Primal Geodesic")
    # visualizer.plot_object(dual_geo, c="blue", label="Dual Geodesic")
    # visualizer.plot_object(
    #    theta_bisector,
    #    alpha=0.7,
    #    c="red",
    #    label="Primal Bisector",
    #    ls="--",
    # )
    visualizer.plot_object(
        eta_bisector,
        alpha=0.7,
        c="blue",
        label="Dual Bisector",
        ls="--",
    )

    visualizer.save(EQT_COORDS, "results/trinomial_chernoff.png")
    visualizer.visualize(EQT_COORDS)


if __name__ == "__main__":
    # main()
    main_chernoff()
