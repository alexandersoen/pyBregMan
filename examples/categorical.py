import numpy as np

from bregman.base import Point
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS
from bregman.visualizer.matplotlib import CoordObjectMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(3)

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

    primal_geo = manifold.theta_geodesic(coord1, coord2)
    dual_geo = manifold.eta_geodesic(coord1, coord2)

    primal_pt = manifold.theta_parallel_transport(coord1, coord2)

    # Define visualizer
    visualizer = CoordObjectMatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(dual_geo, c="red")
    visualizer.animate_object(primal_pt, c="black")

    visualizer.visualize(DISPLAY_TYPE)