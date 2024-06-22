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
                                           Visualize2DTissotIndicatrix)

if __name__ == "__main__":

    # DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = GaussianManifold(1)

    coord1 = Point(LAMBDA_COORDS, np.array([0.0, 1.0]))
    coord2 = Point(LAMBDA_COORDS, np.array([1.0, 1.5]))

    chernoff_point_alpha = ChernoffInformation(manifold).chernoff_point(
        coord1, coord2
    )
    mp = Point(
        ETA_COORDS,
        manifold.convert_coord(ETA_COORDS, coord1).data * 0.5
        + manifold.convert_coord(ETA_COORDS, coord2).data * 0.5,
    )

    print(manifold.convert_coord(ETA_COORDS, coord2))
    print(
        manifold.convert_coord(
            THETA_COORDS, manifold.convert_coord(ETA_COORDS, coord2)
        )
    )

    print(manifold.convert_coord(THETA_COORDS, coord2))
    print(
        manifold.convert_coord(
            ETA_COORDS, manifold.convert_coord(THETA_COORDS, coord2)
        )
    )

    print(
        "Chernoff Information:", ChernoffInformation(manifold)(coord1, coord2)
    )

    primal_geo = BregmanGeodesic(manifold, coord1, coord2, DualCoord.THETA)
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, DualCoord.ETA)

    chernoff_point = primal_geo(1 - chernoff_point_alpha)

    eta_bisector = BregmanBisector(
        manifold,
        coord1,
        coord2,
        coord=DualCoord.ETA,
    )

    theta_bisector = BregmanBisector(
        manifold,
        coord1,
        coord2,
        coord=DualCoord.THETA,
    )

    # Define visualizer
    visualizer = BregmanObjectMatplotlibVisualizer(
        manifold, VISUALIZE_INDEX  # , dim_names=(r"$\mu$", r"$\sigma$")
    )
    metric_cb = Visualize2DTissotIndicatrix()

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
    #    visualizer.plot_object(
    #        theta_bisector, alpha=0.7, c="blue", label="Primal Bisector"
    #    )

    print("Chernoff Check")
    eta_chernoff = manifold.convert_coord(ETA_COORDS, chernoff_point).data
    eta1 = manifold.convert_coord(ETA_COORDS, coord1).data
    eta2 = manifold.convert_coord(ETA_COORDS, coord2).data
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

    # Add animations
    #     visualizer.animate_object(primal_geo, c="blue")
    #     visualizer.animate_object(dual_geo, c="red")

    visualizer.add_callback(metric_cb)

    visualizer.visualize(DISPLAY_TYPE)
