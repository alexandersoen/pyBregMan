import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bregman.manifold.Gaussian1D import Gaussian1DManifold


def make_tangent(p, pt):
    q = p + pt
    return np.vstack([p, q])


if __name__ == "__main__":

    num_frames = 600
    delta = 1 / num_frames

    # 1D Gaussians ordinary parameters (mu, sigma_2)
    coord1 = np.array([-1, 1])
    coord2 = np.array([2, 3])
    #  coord2 = np.array([1, 1])

    manifold = Gaussian1DManifold()

    theta1 = manifold.coord_to_natural(coord1)
    theta2 = manifold.coord_to_natural(coord2)

    primal_geo = manifold.primal_geodesic(theta1, theta2)
    dual_geo = manifold.dual_geodesic(
        manifold.natural_to_moment(theta1), manifold.natural_to_moment(theta2)
    )

    primal_geo_data = np.vstack(
        [primal_geo(delta * t) for t in range(num_frames + 1)]
    )
    dual_geo_data = np.vstack(
        [
            manifold.moment_to_natural(dual_geo(delta * t))
            for t in range(num_frames + 1)
        ]
    )

    primal_tgeo_data = np.vstack(
        [primal_geo.tangent(delta * t) for t in range(num_frames + 1)]
    )
    dual_tgeo_data = np.vstack(
        [
            manifold.moment_to_natural(dual_geo.tangent(delta * t))
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
        *theta1,
        label=r"Source $(\mu = {:.1f}, \sigma^2 = {:.1f})$".format(*coord1),
    )
    ax.scatter(
        *theta2,
        label=r"Target $(\mu = {:.1f}, \sigma^2 = {:.1f})$".format(*coord2),
    )

    prime_pt = ax.scatter(*primal_geo_data[0], c="blue")
    dual_pt = ax.scatter(*dual_geo_data[0], c="red")

    ptn0 = make_tangent(primal_geo_data[0], primal_tgeo_data[0])
    dtn0 = make_tangent(dual_geo_data[0], dual_tgeo_data[0])
    (prime_tn,) = ax.plot(ptn0[:, 0], ptn0[:, 1], c="blue")
    (dual_tn,) = ax.plot(dtn0[:, 0], dtn0[:, 1], c="red")

    def update(frame):
        prime_pt.set_offsets(primal_geo_data[frame])
        dual_pt.set_offsets(dual_geo_data[frame])

        ptnt = make_tangent(primal_geo_data[frame], primal_tgeo_data[frame])
        dtnt = make_tangent(dual_geo_data[frame], dual_tgeo_data[frame])

        prime_tn.set_data(ptnt[:, 0], ptnt[:, 1])
        dual_tn.set_data(dtnt[:, 0], dtnt[:, 1])

        return prime_pt

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=num_frames, interval=60
    )

    ax.legend()
    plt.show()
