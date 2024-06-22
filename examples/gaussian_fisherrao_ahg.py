import numpy as np

from bregman.application.application import LAMBDA_COORDS
from bregman.application.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.base import Point
from bregman.dissimilarity.gaussian import GaussianFisherRaoDistance
from bregman.geodesic.gaussian import KobayashiGeodesic
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, DualCoord
from bregman.visualizer.matplotlib import BregmanObjectMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (2, 3)

    num_frames = 120
    resolution = 120

    # Define manifold + objects
    # manifold = GaussianManifold(1)

    # coord1 = Point(LAMBDA_COORDS, np.array([-1, 1]))
    # coord2 = Point(LAMBDA_COORDS, np.array([1, 1]))

    manifold = GaussianManifold(2)

    coord1 = Point(LAMBDA_COORDS, np.array([0, 0, 1, 0.5, 0.5, 2]))
    coord2 = Point(LAMBDA_COORDS, np.array([0, 0, 1, 0, 0, 0.5]))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, coord=DualCoord.THETA
    )
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, coord=DualCoord.ETA)
    kobayashi = KobayashiGeodesic(manifold, coord1, coord2)

    # Define visualizer
    visualizer = BregmanObjectMatplotlibVisualizer(
        manifold, VISUALIZE_INDEX, resolution=resolution, frames=num_frames
    )

    # Add objects to visualize
    visualizer.plot_object(coord1, c="black")
    visualizer.plot_object(coord2, c="black")
    # visualizer.plot_object(primal_geo(0.5), c="blue")
    # visualizer.plot_object(dual_geo(0.5), c="red")
    visualizer.plot_object(primal_geo, c="blue")
    visualizer.plot_object(kobayashi, c="purple")
    visualizer.plot_object(dual_geo, c="red")

    p, q = coord1, coord2
    ITERS = 5
    for i in range(ITERS):
        primal_geo = BregmanGeodesic(manifold, p, q, coord=DualCoord.THETA)
        dual_geo = BregmanGeodesic(manifold, p, q, coord=DualCoord.ETA)

        if i > 0:
            visualizer.plot_object(p, c="blue", alpha=0.3)
            visualizer.plot_object(q, c="red", alpha=0.3)

            visualizer.plot_object(primal_geo, c="blue", alpha=0.3)
            visualizer.plot_object(dual_geo, c="red", alpha=0.3)

        p = primal_geo(0.5)
        q = dual_geo(0.5)

    visualizer.plot_object(kobayashi(0.5), marker="x", c="purple", zorder=99)

    # cov_cb = VisualizeGaussian2DCovariancePoints()
    # visualizer.add_callback(cov_cb)

    visualizer.visualize(DISPLAY_TYPE)

    fr_dist = GaussianFisherRaoDistance(manifold)
    print("FR Distance", fr_dist(coord1, coord2))
