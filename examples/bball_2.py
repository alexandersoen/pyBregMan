import matplotlib.pyplot as plt
import numpy as np

from bregman.base import Point
from bregman.manifold.ball import KL2DBregmanBallCurve
from bregman.manifold.manifold import THETA_COORDS

if __name__ == "__main__":
    # Bregman ball center and radius
    cx = 6
    cy = 6
    r = 4

    bball = KL2DBregmanBallCurve(Point(THETA_COORDS, np.array([cx, cx])), 4)

    ts = np.linspace(0, 1, 100000)
    print(ts)
    print(bball(0))
    print(bball(0.001))
    print(bball(1))

    datas = np.stack([bball(t).data for t in ts])

    # implicit contour plot
    delta = 0.025
    xrange = np.arange(0.1, 16.5, delta)
    yrange = np.arange(0.1, 16.5, delta)
    X, Y = np.meshgrid(xrange, yrange)
    # F is one side of the equation, G is the other
    F = X - cx + cx * np.log(cx / X)
    G = r - (Y - cy + cy * np.log(cy / Y))
    # parametric contour plot

    x, y = datas[:, 0], datas[:, 1]

    with plt.style.context("bmh"):
        plt.figure(figsize=(6, 6))
        plt.contour(X, Y, (F - G), [0], colors=["tab:blue"], linewidths=[3])
        plt.plot(
            x,
            y,
            ls="--",
            lw=3,
            color="tab:orange",
            label="parametric contour plot",
        )
        plt.legend()
        plt.title(
            r"Bregman Kullback-Leibler ball $c=({},{})$ $\tau={}$".format(
                cx, cy, r
            )
        )
    with plt.style.context("bmh"):
        plt.plot(datas[:, 0], datas[:, 1])
        plt.show()
