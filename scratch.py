import numpy as np

from bregman.base import Point
from bregman.manifold.application import ORDINARY_COORDS
from bregman.manifold.distribution.exponential_family.normal import \
    Gaussian1DManifold
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = ORDINARY_COORDS

    num_frames = 120

    # Define manifold + objects
    manifold = Gaussian1DManifold()

    coord1 = Point(ORDINARY_COORDS, np.array([1, 1]))
    coord2 = Point(ORDINARY_COORDS, np.array([3, 1]))

    primal_geo = manifold.natural_geodesic(coord1, coord2)
    dual_geo = manifold.moment_geodesic(coord1, coord2)

    # Define visualizer
    visualizer = MatplotlibVisualizer(manifold, (0, 1))

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.point_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.point_to_display(coord2))
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
