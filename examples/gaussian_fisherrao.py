import numpy as np

from bregman.base import Point
from bregman.distance.gaussian import GaussianFisherRaoDistance
from bregman.geodesic.gaussian import KobayashiGeodesic
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.manifold.geodesic import BregmanGeodesic
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

    coord1 = Point(LAMBDA_COORDS, np.array([-1.0, 1.0]))
    coord2 = Point(LAMBDA_COORDS, np.array([1.0, 1.0]))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, coord=DualCoord.THETA
    )
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, coord=DualCoord.ETA)
    kobayashi = KobayashiGeodesic(manifold, coord1, coord2)

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

    fr_dist = GaussianFisherRaoDistance(manifold)
    print("FR Distance", fr_dist(coord1, coord2))

    visualizer.visualize(DISPLAY_TYPE)
