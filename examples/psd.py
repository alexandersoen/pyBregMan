# Animated example of geodesics on PSD manifold.

import numpy as np

from bregman.application.psd import PSDManifold
from bregman.base import ETA_COORDS, LAMBDA_COORDS, DualCoords, Point
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = PSDManifold(2)

    _m1 = np.array([[1, 3], [3, 1]])
    _m2 = np.array([[2, 1], [1, 2]])

    m1 = (_m1 @ _m1.T)[np.triu_indices(2)]
    m2 = (_m2 @ _m2.T)[np.triu_indices(2)]

    coord1 = Point(LAMBDA_COORDS, m1)
    coord2 = Point(LAMBDA_COORDS, m2)

    print(m1)
    print(manifold.theta_generator(coord1.data))
    print(manifold.theta_generator.grad(coord1.data))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.THETA
    )
    dual_geo = BregmanGeodesic(
        manifold, coord1, coord2, dcoords=DualCoords.ETA
    )

    # Define visualizer
    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="red", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="blue", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="red")
    visualizer.animate_object(dual_geo, c="blue")

    visualizer.visualize(DISPLAY_TYPE)
