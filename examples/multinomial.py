from pathlib import Path

import numpy as np

from bregman.application.distribution.exponential_family.multinomial import \
    MultinomialManifold
from bregman.base import (ETA_COORDS, LAMBDA_COORDS, THETA_COORDS, DualCoords,
                          Point)
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.visualizer.matplotlib import BregmanObjectMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = MultinomialManifold(3, 100)

    p1 = np.array([0.2, 0.4, 0.4])
    p2 = np.array([0.3, 0.5, 0.2])

    coord1 = Point(LAMBDA_COORDS, p1)
    coord2 = Point(LAMBDA_COORDS, p2)

    print(coord1)
    print(manifold.convert_coord(THETA_COORDS, coord1))
    print(
        manifold.convert_coord(
            LAMBDA_COORDS, manifold.convert_coord(THETA_COORDS, coord1)
        )
    )

    print(coord1)
    print(manifold.convert_coord(ETA_COORDS, coord1))
    print(
        manifold.convert_coord(
            LAMBDA_COORDS, manifold.convert_coord(ETA_COORDS, coord1)
        )
    )

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.THETA
    )
    dual_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.ETA
    )

    # Define visualizer
    visualizer = BregmanObjectMatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(dual_geo, c="red")

    # visualizer.visualize(DISPLAY_TYPE)
    SAVE_PATH = Path("figures/multinomial.pdf")
    visualizer.save(DISPLAY_TYPE, SAVE_PATH)
