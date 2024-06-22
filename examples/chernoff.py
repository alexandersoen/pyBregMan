import numpy as np

from bregman.base import Point
from bregman.distance.bregman import ChernoffInformation
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, DualCoord
from bregman.visualizer.matplotlib import BregmanObjectMatplotlibVisualizer

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(2)

    coord1 = Point(LAMBDA_COORDS, np.array([0.7, 0.3]))
    coord2 = Point(LAMBDA_COORDS, np.array([0.1, 0.9]))

    chernoff_information = ChernoffInformation(manifold, eps=1e-10)
    chernoff_point_alpha = chernoff_information.chernoff_point(coord1, coord2)
    mp = Point(
        ETA_COORDS,
        manifold.convert_coord(ETA_COORDS, coord1).data * 0.5
        + manifold.convert_coord(ETA_COORDS, coord2).data * 0.5,
    )

    print("Chernoff Information:", chernoff_information(coord1, coord2))

    primal_geo = BregmanGeodesic(
        manifold, coord1, coord2, coord=DualCoord.THETA
    )
    dual_geo = BregmanGeodesic(manifold, coord1, coord2, coord=DualCoord.ETA)

    chernoff_point = primal_geo(1 - chernoff_point_alpha)

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

    visualizer.visualize(DISPLAY_TYPE)

    # Generator plot for chernoff information
    import matplotlib.pyplot as plt

    dcoord = DualCoord.ETA

    p_vals = np.arange(0.01, 0.99, 0.01)
    plot_points = [Point(LAMBDA_COORDS, np.array([p, 1 - p])) for p in p_vals]
    coord_points = [
        manifold.convert_coord(dcoord.value, p) for p in plot_points
    ]
    coord_plot_vals = [coord.data.item() for coord in coord_points]
    gen_plot_vals = [
        manifold.bregman_generator(dcoord)(coord.data.item())
        for coord in coord_points
    ]
    plt.plot(coord_plot_vals, gen_plot_vals)

    coord1_pv = [
        manifold.convert_coord(dcoord.value, coord1).data.item(),
        manifold.bregman_generator(dcoord)(
            manifold.convert_coord(dcoord.value, coord1).data.item()
        ),
    ]
    coord2_pv = [
        manifold.convert_coord(dcoord.value, coord2).data.item(),
        manifold.bregman_generator(dcoord)(
            manifold.convert_coord(dcoord.value, coord2).data.item()
        ),
    ]
    chernoff_point_pv = [
        manifold.convert_coord(dcoord.value, chernoff_point).data.item(),
        manifold.bregman_generator(dcoord)(
            manifold.convert_coord(dcoord.value, chernoff_point).data.item()
        ),
    ]

    plt.scatter(*coord1_pv, c="red")
    plt.scatter(*coord2_pv, c="blue")
    plt.scatter(*chernoff_point_pv, c="purple")
    plt.plot([coord1_pv[0], coord2_pv[0]], [coord1_pv[1], coord2_pv[1]])

    print("test", np.min(gen_plot_vals))

    for x in [coord1_pv[0], coord2_pv[0], chernoff_point_pv[0]]:
        plt.vlines(
            [x], [np.min(gen_plot_vals)], [np.max(gen_plot_vals)], alpha=0.3
        )

    plt.show()
