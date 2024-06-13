from abc import ABC, abstractmethod

import numpy as np
from scipy.special import lambertw

from bregman.base import CoordObject, Curve, Point
from bregman.manifold.manifold import THETA_COORDS, BregmanManifold, DualCoord


class BregmanBall(CoordObject, ABC):

    center: Point
    radius: float
    dual_coords: DualCoord

    @abstractmethod
    def is_in(self, other: Point) -> bool:
        pass

    def parametrized_curve(self) -> Curve:
        return NotImplemented()


class GeneratorBregmanBall(BregmanBall):

    def __init__(
        self,
        manifold: BregmanManifold,
        center: Point,
        radius: float,
        coords: DualCoord = DualCoord.THETA,
    ) -> None:
        super().__init__(coords.value)

        self.manifold = manifold

        self.center = center
        self.radius = radius

        self.dual_coords = coords

    def is_in(self, other: Point) -> bool:
        return (
            self.manifold.bregman_divergence(self.center, other).item()
            < self.radius
        )


def bregman_badoiu_clarkson(
    manifold: BregmanManifold,
    points: list[Point],
    T: int,
    coords: DualCoord = DualCoord.THETA,
) -> GeneratorBregmanBall:
    divergence = manifold.bregman_generator(coords).bergman_divergence
    primal_gen = manifold.bregman_generator(coords)
    dual_gen = manifold.bregman_generator(coords.dual())

    datas = [p.data for p in points]

    c = datas[0]
    d = 0
    for t in range(1, T):
        dist_pairs = [(p, divergence(c, p).item()) for p in datas]
        s, d = min(dist_pairs, key=lambda pair: pair[1])
        c = dual_gen(t / (t + 1) * primal_gen(c) + 1 / (t + 1) * primal_gen(s))

    center = Point(coords=coords.value, data=c)
    return GeneratorBregmanBall(manifold, center, d, coords)


class KL2DBregmanBallCurve(Curve):

    def __init__(self, center: Point, radius: float) -> None:
        assert len(center.data) == 2

        super().__init__(coords=THETA_COORDS)

        self.center = center
        self.radius = radius

    def path(self, t: float) -> Point:
        """Path from top right quadrant counter-clockwise"""
        cx, cy = self.center.data

        if t < 0.25:
            # Top right
            u = (1 - t / 0.25) * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=-1))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=-1)
            )
        elif t < 0.5:
            # Top Left
            u = (t - 0.25) / 0.25 * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=0))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=-1)
            )
        elif t < 0.75:
            # Bot Left
            u = (1 - (t - 0.5) / 0.25) * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=0))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=0)
            )
        else:
            # Bot Right
            u = (t - 0.75) / 0.25 * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=-1))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=0)
            )

        return Point(data=np.array([x, y]), coords=self.coords)
