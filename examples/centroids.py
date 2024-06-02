import numpy as np

from bregman.base import Point
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS
from bregman.visualizer.matplotlib import MatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(3)

    p1 = np.array([0.2, 0.4, 0.4])
    p2 = np.array([0.3, 0.5, 0.2])
    p3 = np.array([0.7, 0.1, 0.2])

    p1 = Point(LAMBDA_COORDS, p1)
    p2 = Point(LAMBDA_COORDS, p2)
    p3 = Point(LAMBDA_COORDS, p3)

    points = [p1, p2, p3]

    # Triangles
    p12_primal_geo = manifold.theta_geodesic(p1, p2)
    p13_primal_geo = manifold.theta_geodesic(p1, p3)
    p23_primal_geo = manifold.theta_geodesic(p2, p3)
    p12_dual_geo = manifold.eta_geodesic(p1, p2)
    p13_dual_geo = manifold.eta_geodesic(p1, p3)
    p23_dual_geo = manifold.eta_geodesic(p2, p3)

    # Centroids
    alphas = [0.5, 0.5, 0.5]
    weights = [1.0, 1.0, 1.0]

    js_centroid = manifold.skew_burbea_rao_barycenter(points, alphas, weights)
    theta_centroid = manifold.theta_barycenter(points, weights)
    eta_centroid = manifold.eta_barycenter(points, weights)

    # Define visualizer
    visualizer = MatplotlibVisualizer(manifold, VISUALIZE_INDEX)

    # Add objects to visualize
    visualizer.plot_object(p1, label=manifold.convert_to_display(p1))
    visualizer.plot_object(p2, label=manifold.convert_to_display(p2))
    visualizer.plot_object(p3, label=manifold.convert_to_display(p3))

    visualizer.plot_object(
        theta_centroid,
        c="blue",
        marker="x",
        label=f"Theta Centroid: {manifold.convert_to_display(theta_centroid)}",
    )
    visualizer.plot_object(
        js_centroid,
        c="purple",
        marker="x",
        label=f"JS Centroid: {manifold.convert_to_display(js_centroid)}",
    )
    visualizer.plot_object(
        eta_centroid,
        c="red",
        marker="x",
        label=f"Eta Centroid: {manifold.convert_to_display(eta_centroid)}",
    )

    visualizer.plot_object(p12_primal_geo, c="blue")
    visualizer.plot_object(p13_primal_geo, c="blue")
    visualizer.plot_object(p23_primal_geo, c="blue")

    visualizer.plot_object(p12_dual_geo, c="red")
    visualizer.plot_object(p13_dual_geo, c="red")
    visualizer.plot_object(p23_dual_geo, c="red")

    visualizer.visualize(DISPLAY_TYPE)
