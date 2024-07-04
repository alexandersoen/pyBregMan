# Generate pyBregMan logo from different geodesics on the Gaussian
# manifold.

import numpy as np

from bregman.application.distribution.exponential_family.gaussian import (
    FisherRaoKobayashiGeodesic, GaussianFisherRaoDistance, GaussianManifold)
from bregman.base import (ETA_COORDS, LAMBDA_COORDS, THETA_COORDS, DualCoords,
                          Point)
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = GaussianManifold(1)

    coord1 = Point(LAMBDA_COORDS, np.array([-1.0, 1.0]))
    coord2 = Point(LAMBDA_COORDS, np.array([1.0, 1.0]))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.THETA
    )
    dual_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.ETA
    )
    kobayashi = FisherRaoKobayashiGeodesic(manifold, coord1, coord2)

    # Define visualizer
    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, c="black")
    visualizer.plot_object(coord2, c="black")
    visualizer.plot_object(primal_geo(0.5), c="red")
    visualizer.plot_object(kobayashi(0.5), c="purple")
    visualizer.plot_object(dual_geo(0.5), c="blue")
    visualizer.plot_object(primal_geo, c="red")
    visualizer.plot_object(kobayashi, c="purple")
    visualizer.plot_object(dual_geo, c="blue")

    fr_dist = GaussianFisherRaoDistance(manifold)
    print("FR Distance", fr_dist(coord1, coord2))

    visualizer.visualize(DISPLAY_TYPE)
