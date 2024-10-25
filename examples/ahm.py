# Inductive arithmetic-harmonic-mean converging to the geometric matrix
# mean. Usually, the algorithm only works on the PSD manifold.
# However, we can example this algorithm on the Gaussian manifold when
# we ensure that the points are centered.

import jax.numpy as jnp

from bregman.application.distribution.exponential_family.gaussian import (
    FisherRaoKobayashiGeodesic,
    GaussianManifold,
)
from bregman.base import (
    ETA_COORDS,
    LAMBDA_COORDS,
    THETA_COORDS,
    DualCoords,
    Point,
)
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.visualizer.matplotlib import MultiMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    VISUALIZE_INDEX = (2, 3)

    coords = [LAMBDA_COORDS, ETA_COORDS, THETA_COORDS]

    num_frames = 120
    resolution = 120

    manifold = GaussianManifold(2)

    coord1 = Point(LAMBDA_COORDS, jnp.array([0, 0, 1, 0.5, 0.5, 2]))
    coord2 = Point(LAMBDA_COORDS, jnp.array([0, 0, 1, 0, 0, 0.5]))

    primal_geo = BregmanGeodesic(manifold, coord1, coord2, dcoords=DualCoords.THETA)
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, dcoords=DualCoords.ETA)
    kobayashi = FisherRaoKobayashiGeodesic(manifold, coord1, coord2)

    # manifold, VISUALIZE_INDEX, resolution=resolution, frames=num_frames

    # Define visualizer
    multivisualizer = MultiMatplotlibVisualizer(
        nrows=1, ncols=3, resolution=resolution, frames=num_frames
    )
    visualizer = multivisualizer.new_visualizer(
        0, 0, manifold, coords[0], plot_dims=VISUALIZE_INDEX
    )

    # Add objects to visualize
    visualizer.plot_object(coord1, c="black")
    visualizer.plot_object(coord2, c="black")
    visualizer.plot_object(primal_geo, c="red", label="Primal Geodesics")
    visualizer.plot_object(kobayashi, c="purple", label="Dual Geodesics")
    visualizer.plot_object(dual_geo, c="blue")

    p, q = coord1, coord2
    ITERS = 2
    for i in range(ITERS):
        primal_geo = BregmanGeodesic(manifold, p, q, dcoords=DualCoords.THETA)
        dual_geo = BregmanGeodesic(manifold, p, q, dcoords=DualCoords.ETA)

        if i > 0:
            visualizer.plot_object(p, c="red", alpha=0.3)
            visualizer.plot_object(q, c="blue", alpha=0.3)

            visualizer.plot_object(primal_geo, c="red", alpha=0.3)
            visualizer.plot_object(dual_geo, c="blue", alpha=0.3)

        p = primal_geo(0.5)
        q = dual_geo(0.5)

    visualizer.plot_object(
        kobayashi(0.5), marker="x", c="purple", zorder=99, label="FR Centroid"
    )

    multivisualizer.copy_visualizer(0, 0, 0, 1, coords[1])
    multivisualizer.copy_visualizer(0, 0, 0, 2, coords[2])

    multivisualizer.visualize_all()
