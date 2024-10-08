# Ad-hoc method for viewing Bregman balls.

import numpy as np

from bregman.application.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.base import LAMBDA_COORDS, THETA_COORDS, DualCoords, Point
from bregman.dissimilarity.bregman import BregmanDivergence

if __name__ == "__main__":

    DISPLAY_TYPE = LAMBDA_COORDS
    VISUALIZE_INDEX = (0, 1)

    num_frames = 120

    # Define manifold + objects
    manifold = CategoricalManifold(3)

    c1 = np.array([0.2, 0.4, 0.4])
    c1 = Point(LAMBDA_COORDS, c1)

    r = 0.2
    eps = 1e-3

    xs = np.arange(0, 1, 1 / 200)[1:-1]
    ys = np.arange(0, 1, 1 / 200)[1:-1]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax = plt.figure().add_subplot(projection="3d")

    xs, ys = np.meshgrid(xs, ys)

    zs = np.zeros_like(xs)
    ws = np.zeros_like(xs)
    bds = np.zeros_like(xs)

    lines = []

    lift = []
    breg_divergence = BregmanDivergence(manifold)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            x = xs[i, j]
            y = ys[i, j]
            if x + y > 1 - eps:
                zs[i, j] = None
                ws[i, j] = None
                bds[i, j] = None
                continue

            test_point = Point(LAMBDA_COORDS, np.array([x, y, 1 - x - y]))
            test_f = manifold.bregman_generator(dcoords=DualCoords.THETA)(
                manifold.convert_coord(THETA_COORDS, test_point).data
            )

            test_bd = breg_divergence(test_point, c1)
            if test_bd < r:
                bds[i, j] = 0
                ws[i, j] = test_f
                zs[i, j] = None
                if test_bd > r - 1e-2:
                    lines.append([[x, y, 0], [x, y, test_f]])
            else:
                bds[i, j] = None
                ws[i, j] = None
                zs[i, j] = test_f

    ax.plot_surface(
        xs,
        ys,
        zs,
        edgecolor="blue",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )
    ax.add_collection3d(
        Poly3DCollection(
            lines,
            edgecolor="red",
            lw=0.5,
            alpha=0.3,
        ),
    )

    ax.plot_surface(
        xs,
        ys,
        ws,
        edgecolor="red",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )
    ax.plot_surface(
        xs,
        ys,
        bds,
        edgecolor="red",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )
    ax.contourf(xs, ys, zs, zdir="z", offset=-100, cmap="coolwarm")
    ax.contourf(xs, ys, zs, zdir="x", offset=-40, cmap="coolwarm")
    ax.contourf(xs, ys, zs, zdir="y", offset=40, cmap="coolwarm")

    plt.show()
