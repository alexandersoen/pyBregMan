import numpy as np

from bregman.base import Point
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.manifold import ETA_COORDS
from bregman.manifold.psd import PSDManifold
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

    primal_geo = manifold.theta_geodesic(coord1, coord2)
    dual_geo = manifold.eta_geodesic(coord1, coord2)

    # Define visualizer
    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(coord1, label=manifold.convert_to_display(coord1))
    visualizer.plot_object(coord2, label=manifold.convert_to_display(coord2))
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    # Add animations
    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
