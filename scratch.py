import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bregman.base import CoordType, Point
from bregman.manifold.normal import Gaussian1DManifold


def make_tangent(p, pt):
    q = p + pt
    return np.vstack([p, q])


if __name__ == "__main__":

    # DISPLAY_TYPE = CoordType.MOMENT
    # DISPLAY_TYPE = CoordType.NATURAL
    DISPLAY_TYPE = CoordType.LAMBDA

    num_frames = 120
    delta = 1 / num_frames

    # 1D Gaussians ordinary parameters (mu, sigma_2)
    # coord1 = Point(CoordType.LAMBDA, np.array([-1, 1]))
    # coord2 = Point(CoordType.LAMBDA, np.array([2, 3]))

    coord1 = Point(CoordType.LAMBDA, np.array([1, 1]))
    coord2 = Point(CoordType.LAMBDA, np.array([3, 1]))
    ###########

    manifold = Gaussian1DManifold()

    theta1 = manifold.transform_coord(CoordType.NATURAL, coord1)
    theta2 = manifold.transform_coord(CoordType.NATURAL, coord2)

    eta1 = manifold.transform_coord(CoordType.MOMENT, coord1)
    eta2 = manifold.transform_coord(CoordType.MOMENT, coord2)

    primal_geo = manifold.primal_geodesic(theta1, theta2)
    dual_geo = manifold.dual_geodesic(eta1, eta2)

    primal_geo_data = np.vstack(
        [
            manifold.transform_coord(DISPLAY_TYPE, primal_geo(delta * t)).coord
            for t in range(num_frames + 1)
        ]
    )
    dual_geo_data = np.vstack(
        [
            manifold.transform_coord(DISPLAY_TYPE, dual_geo(delta * t)).coord
            for t in range(num_frames + 1)
        ]
    )

    fig, ax = plt.subplots()

    ax.plot(
        primal_geo_data[:, 0],
        primal_geo_data[:, 1],
        c="blue",
        ls="--",
        linewidth=1,
    )

    ax.plot(
        dual_geo_data[:, 0],
        dual_geo_data[:, 1],
        c="red",
        ls="--",
        linewidth=1,
    )

    ax.scatter(
        *manifold.transform_coord(DISPLAY_TYPE, theta1).coord,
        label=r"Source $(\mu = {:.1f}, \sigma^2 = {:.1f})$".format(
            *manifold.transform_coord(CoordType.LAMBDA, theta1).coord
        ),
    )
    ax.scatter(
        *manifold.transform_coord(DISPLAY_TYPE, theta2).coord,
        label=r"Target $(\mu = {:.1f}, \sigma^2 = {:.1f})$".format(
            *manifold.transform_coord(CoordType.LAMBDA, theta2).coord
        ),
    )

    prime_pt = ax.scatter(*primal_geo_data[0], c="blue")
    dual_pt = ax.scatter(*dual_geo_data[0], c="red")

    def update(frame):
        prime_pt.set_offsets(primal_geo_data[frame])
        dual_pt.set_offsets(dual_geo_data[frame])

        return prime_pt

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=num_frames, interval=1
    )

    ax.legend()
    plt.show()
