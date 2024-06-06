import numpy as np

from bregman.base import Point
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.gaussian import (
    EriksenIVPGeodesic, GaussianManifold)
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, DualCoord
from bregman.visualizer.matplotlib import CoordObjectMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = GaussianManifold(2)

    coord1 = Point(LAMBDA_COORDS, np.array([0, 0, 1, 0, 0, 1]))
    coord2 = Point(LAMBDA_COORDS, np.array([1, 2, 1, 0, 0, 0.5]))

    mp = Point(
        ETA_COORDS,
        manifold.convert_coord(ETA_COORDS, coord1).data * 0.5
        + manifold.convert_coord(ETA_COORDS, coord2).data * 0.5,
    )

    print(
        "Chernoff Information:", manifold.chernoff_information(coord1, coord2)
    )

    primal_geo = manifold.bregman_geodesic(coord1, coord2, DualCoord.THETA)
    dual_geo = manifold.bregman_geodesic(coord1, coord2, DualCoord.ETA)

    eriksen = EriksenIVPGeodesic(coord2, manifold)

    print(eriksen(0))
    print(eriksen(0.5))
    print(eriksen(1))

    # Define visualizer
    visualizer = CoordObjectMatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(eriksen, c="purple", label="Eriksen Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(eriksen, c="purple")
    visualizer.animate_object(dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
