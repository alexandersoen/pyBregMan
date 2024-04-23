import numpy as np

from bregman.base import CoordType, Point
from bregman.manifold.normal import Gaussian1DManifold
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = CoordType.LAMBDA

    num_frames = 120

    manifold = Gaussian1DManifold()

    coord1 = Point(CoordType.LAMBDA, np.array([1, 1]))
    coord2 = Point(CoordType.LAMBDA, np.array([3, 1]))

    primal_geo = manifold.primal_geodesic(coord1, coord2)
    dual_geo = manifold.dual_geodesic(coord1, coord2)

    visualizer = MatplotlibVisualizer(manifold, (0, 1))

    visualizer.plot_object(
        coord1,
        label=r"Source $(\mu = {:.1f}, \sigma^2 = {:.1f})$".format(
            *manifold.transform_coord(CoordType.LAMBDA, coord1).coord
        ),
    )
    visualizer.plot_object(
        coord2,
        label=r"Source $(\mu = {:.1f}, \sigma^2 = {:.1f})$".format(
            *manifold.transform_coord(CoordType.LAMBDA, coord2).coord
        ),
    )
    visualizer.plot_object(primal_geo, c="blue", label="Primal Geodesic")
    visualizer.plot_object(dual_geo, c="red", label="Dual Geodesic")

    visualizer.animate_object(primal_geo, c="blue")
    visualizer.animate_object(dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
