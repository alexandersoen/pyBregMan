import numpy as np

from bregman.base import Point
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.gaussian import (
    GaussianManifold, KobayashiGeodesic)
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, DualCoord
from bregman.visualizer.matplotlib import CoordObjectMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = GaussianManifold(1)

    coord1 = Point(LAMBDA_COORDS, np.array([-1, 1]))
    coord2 = Point(LAMBDA_COORDS, np.array([1, 1]))

    primal_geo = manifold.bregman_geodesic(coord1, coord2, DualCoord.THETA)
    dual_geo = manifold.bregman_geodesic(coord1, coord2, DualCoord.ETA)
    kobayashi = KobayashiGeodesic(coord1, coord2, manifold)

    # Define visualizer
    visualizer = CoordObjectMatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, c="black")
    visualizer.plot_object(coord2, c="black")
    visualizer.plot_object(primal_geo(0.5), c="blue")
    visualizer.plot_object(kobayashi(0.5), c="purple")
    visualizer.plot_object(dual_geo(0.5), c="red")
    visualizer.plot_object(primal_geo, c="blue")
    visualizer.plot_object(kobayashi, c="purple")
    visualizer.plot_object(dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
