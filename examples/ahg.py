import numpy as np

from bregman.base import Point
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.manifold import ETA_COORDS
from bregman.manifold.psd import PSDManifold
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    ITERS = 3

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

    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    p, q = coord1, coord2
    for i in range(ITERS):
        primal_geo = manifold.theta_geodesic(p, q)
        dual_geo = manifold.eta_geodesic(p, q)

        if i == 0:
            visualizer.plot_object(
                p, c="green", label=manifold.convert_to_display(p)
            )
            visualizer.plot_object(
                q, c="orange", label=manifold.convert_to_display(q)
            )

            visualizer.plot_object(
                primal_geo, c="blue", label="Primal Geodesic"
            )
            visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")
        else:
            visualizer.plot_object(p, c="green")
            visualizer.plot_object(q, c="orange")

            visualizer.plot_object(primal_geo, c="blue")
            visualizer.plot_object(dual_geo, c="red")

        p = primal_geo(0.5)
        q = dual_geo(0.5)

    visualizer.visualize(DISPLAY_TYPE)