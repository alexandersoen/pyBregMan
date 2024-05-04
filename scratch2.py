import numpy as np

from bregman.base import Point
from bregman.manifold.application import ORDINARY_COORDS
from bregman.manifold.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = ORDINARY_COORDS
    VISUALIZE_INDEX = (2, 5)

    num_frames = 120

    # Define manifold + objects
    manifold = GaussianManifold(2)

    coord1 = Point(ORDINARY_COORDS, np.array([0, 0, 1, 0, 0, 1]))
    coord2 = Point(ORDINARY_COORDS, np.array([0, 0, 2, 0, 0, 2]))

    primal_geo = manifold.natural_geodesic(coord1, coord2)
    dual_geo = manifold.moment_geodesic(coord1, coord2)

    # Define visualizer
    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.point_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.point_to_display(coord2))
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
