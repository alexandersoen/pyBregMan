# Inductive arithmetic-harmonic-mean converging to the geometric matrix mean.
import jax.numpy as jnp

from bregman.application.psd import PSDManifold
from bregman.base import (
    ETA_COORDS,
    LAMBDA_COORDS,
    THETA_COORDS,
    DualCoords,
    Point,
)
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120
    resolution = 120

    manifold = PSDManifold(2)

    coord1 = Point(LAMBDA_COORDS, jnp.array([1, 0.5, 2]))
    coord2 = Point(LAMBDA_COORDS, jnp.array([1, 0, 0.5]))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.THETA
    )
    dual_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.ETA
    )

    # Define visualizer
    visualizer = MatplotlibVisualizer(
        manifold, VISUALIZE_INDEX, resolution=resolution, frames=num_frames
    )

    # Add objects to visualize
    visualizer.plot_object(coord1, c="black", label="Initial Points")
    visualizer.plot_object(coord2, c="black")
    visualizer.plot_object(primal_geo, c="red", label="Primal Geodesics")
    visualizer.plot_object(dual_geo, c="blue", label="Dual Geodesics")

    p, q = coord1, coord2
    ITERS = 5
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

    visualizer.plot_object(p, c="purple", label="Approximate Geometric Mean")

    visualizer.visualize(DISPLAY_TYPE)
