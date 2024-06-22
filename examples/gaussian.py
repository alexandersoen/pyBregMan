import numpy as np

from bregman.base import Point
from bregman.distance.bregman import ChernoffInformation
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.bisector import BregmanBisector
from bregman.manifold.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, DualCoord
from bregman.visualizer.matplotlib import (BregmanObjectMatplotlibVisualizer,
                                           VisualizeGaussian2DCovariancePoints)

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = GaussianManifold(2)

    coord1 = Point(LAMBDA_COORDS, np.array([0, 0, 1, 0.5, 0.5, 2]))
    coord2 = Point(LAMBDA_COORDS, np.array([1, 2, 1, 0, 0, 0.5]))

    print(manifold.convert_coord(ETA_COORDS, coord2))
    print(
        manifold.convert_coord(
            THETA_COORDS, manifold.convert_coord(ETA_COORDS, coord2)
        )
    )

    chernoff_information = ChernoffInformation(manifold, eps=1e-10)
    chernoff_point_alpha = chernoff_information.chernoff_point(coord1, coord2)
    mp = Point(
        ETA_COORDS,
        manifold.convert_coord(ETA_COORDS, coord1).data * 0.5
        + manifold.convert_coord(ETA_COORDS, coord2).data * 0.5,
    )

    print("Chernoff Information:", chernoff_information(coord1, coord2))

    primal_geo = BregmanGeodesic(manifold, coord1, coord2, DualCoord.THETA)
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, DualCoord.ETA)

    chernoff_point = primal_geo(1 - chernoff_point_alpha)

    # This does not make sense to plot here. Only for 2D
    eta_bisector = BregmanBisector(
        ETA_COORDS,
        manifold.convert_coord(ETA_COORDS, coord1),
        manifold.convert_coord(ETA_COORDS, coord2),
        manifold.bregman_generator(DualCoord.ETA),
    )

    theta_bisector = BregmanBisector(
        THETA_COORDS,
        manifold.convert_coord(THETA_COORDS, coord1),
        manifold.convert_coord(THETA_COORDS, coord2),
        manifold.bregman_generator(DualCoord.THETA),
    )

    # Define visualizer
    visualizer = BregmanObjectMatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(
        chernoff_point,
        label=f"Chernoff Point, alpha={chernoff_point_alpha:.2f}",
    )
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")
    visualizer.plot_object(
        eta_bisector, alpha=0.7, c="red", label="Dual Bisector"
    )
    visualizer.plot_object(
        theta_bisector, alpha=0.7, c="blue", label="Primal Bisector"
    )

    print()
    print("Chernoff Check")
    eta_chernoff = manifold.convert_coord(ETA_COORDS, chernoff_point).data
    eta1 = manifold.convert_coord(ETA_COORDS, coord1).data
    eta2 = manifold.convert_coord(ETA_COORDS, coord2).data
    print(eta1, eta_chernoff, eta2)
    print(
        np.dot(
            eta_chernoff,
            manifold.eta_generator.grad(eta1)
            - manifold.eta_generator.grad(eta2),
        )
        + manifold.eta_generator(eta1)
        - manifold.eta_generator(eta2)
        - (
            np.dot(eta1.data, manifold.eta_generator.grad(eta1))
            - np.dot(eta2.data, manifold.eta_generator.grad(eta2))
        )
    )
    print()

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    # visualizer.animate_object(eriksen, c="purple")
    visualizer.animate_object(dual_geo, c="red")

    cov_cb = VisualizeGaussian2DCovariancePoints()
    visualizer.add_callback(cov_cb)

    visualizer.visualize(DISPLAY_TYPE)
