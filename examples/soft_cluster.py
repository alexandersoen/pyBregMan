import matplotlib.pyplot as plt
import numpy as np

from bregman.base import Point
from bregman.distance.bregman import BregmanDivergence
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.manifold.distribution.mixture.ef_mixture import EFMixtureManifold
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, DualCoord

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    # DISPLAY_TYPE = ETA_COORDS
    # DISPLAY_TYPE = THETA_COORDS
    VISUALIZE_INDEX = (0, 1)

    DIM = 2
    N = 200

    data = np.random.randn(N, DIM)  # N x 2
    mixing = np.random.binomial(1, 0.5, (N, 1))
    data = mixing * np.random.normal(-3.0, size=(N, DIM)) + (
        1 - mixing
    ) * np.random.normal(1.5, size=(N, DIM))

    def make_random_point():
        return Point(
            LAMBDA_COORDS,
            np.concatenate(
                [np.random.randn(DIM), np.eye(DIM).flatten()]
            ).flatten(),
        )

    init_dist_points = [make_random_point(), make_random_point()]
    init_mixing_point = Point(LAMBDA_COORDS, np.array([0.5, 0.5]))

    gaussian_manifold = GaussianManifold(DIM)
    ef_mixture_manifold = EFMixtureManifold(
        [gaussian_manifold.point_to_distribution(p) for p in init_dist_points],
        gaussian_manifold,
    )
    eta_gaussian_div = BregmanDivergence(
        ef_mixture_manifold.ef_manifold, coord=DualCoord.ETA
    )

    def loglikelihood(data, mixing_point, dist_points):
        total = np.zeros(data.shape[0])
        weights = ef_mixture_manifold.convert_coord(
            LAMBDA_COORDS, mixing_point
        ).data
        for w, d in zip(weights, dist_points):
            total += w * np.apply_along_axis(
                ef_mixture_manifold.ef_manifold.point_to_distribution(d).pdf,
                1,
                data,
            )
        return np.sum(np.log(total))

    # For plotting
    mixing_hist = [init_mixing_point]
    mixing_dist = [init_dist_points]

    # Init clustering
    max_T = 50
    t = 0

    cur_mixing_point = init_mixing_point
    cur_dist_points = init_dist_points
    cur_ll = loglikelihood(data, init_mixing_point, init_dist_points)
    old_ll = float("inf")
    thresh_ll = cur_ll * 0.001

    # Shift data point into expected parameters
    def embed(x):
        p = Point(
            LAMBDA_COORDS, np.concatenate([x, np.eye(DIM).flatten()]).flatten()
        )
        eta_p = ef_mixture_manifold.ef_manifold.convert_coord(ETA_COORDS, p)
        return eta_p

    embedded_data = [embed(x) for x in data]
    while t < max_T and abs(cur_ll - old_ll) > thresh_ll:

        # Step E
        div_weights = np.zeros(
            (data.shape[0], ef_mixture_manifold.dimension + 1)
        )

        cur_weights = ef_mixture_manifold.convert_coord(
            LAMBDA_COORDS, cur_mixing_point
        ).data
        for i, (w, d) in enumerate(zip(cur_weights, cur_dist_points)):

            div_values = np.stack(
                [eta_gaussian_div(x, d) for x in embedded_data]
            )
            div_weights[:, i] += w * np.exp(-div_values)

        # Step M
        div_weights = div_weights / np.sum(div_weights, axis=1).reshape(-1, 1)
        new_weights = div_weights.mean(axis=0)
        cur_mixing_point = Point(LAMBDA_COORDS, new_weights)

        cur_dist_points = [
            ef_mixture_manifold.ef_manifold.bregman_barycenter(
                embedded_data, weights=div_weights[:, k], coord=DualCoord.ETA
            )
            for k in range(ef_mixture_manifold.dimension + 1)
        ]

        mixing_hist.append(cur_mixing_point)
        mixing_dist.append(cur_dist_points)

        old_ll = cur_ll
        cur_ll = loglikelihood(data, cur_mixing_point, cur_dist_points)
        t += 1

    # ======================================================
    # PLOTTING TO CHECK
    plt.scatter(
        data[mixing.flatten() == 0, 0],
        np.zeros_like(data[mixing.flatten() == 0, 0]),
        c="orange",
    )
    plt.scatter(
        data[mixing.flatten() == 1, 0],
        np.zeros_like(data[mixing.flatten() == 1, 0]),
        c="purple",
    )
    plt.plot(
        [
            ef_mixture_manifold.ef_manifold.convert_coord(
                LAMBDA_COORDS, ds[0]
            ).data[0]
            for ds in mixing_dist
        ],
        1 + np.arange(len(mixing_dist)),
        c="red",
    )
    plt.plot(
        [
            ef_mixture_manifold.ef_manifold.convert_coord(
                LAMBDA_COORDS, ds[1]
            ).data[0]
            for ds in mixing_dist
        ],
        -1 - np.arange(len(mixing_dist)),
        c="blue",
    )

    plt.vlines([-3, 1.5], -1, 1)
    plt.show()
