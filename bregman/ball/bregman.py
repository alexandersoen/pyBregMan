from bregman.ball.base import Ball
from bregman.base import Point
from bregman.dissimilarity.bregman import BregmanDivergence
from bregman.manifold.manifold import BregmanManifold, DualCoord


class BregmanBall(Ball):

    def __init__(
        self,
        manifold: BregmanManifold,
        center: Point,
        radius: float,
        coord: DualCoord = DualCoord.THETA,
    ) -> None:
        super().__init__(manifold, center, radius, coord.value)

        self.bregman_divergence = BregmanDivergence(manifold, coord=coord)

    def is_in(self, other: Point) -> bool:
        return self.bregman_divergence(self.center, other).item() < self.radius


def bregman_badoiu_clarkson(
    manifold: BregmanManifold,
    points: list[Point],
    T: int,
    coords: DualCoord = DualCoord.THETA,
) -> BregmanBall:
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
    return BregmanBall(manifold, center, d, coords)
