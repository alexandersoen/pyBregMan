from bregman.ball.base import Ball
from bregman.base import DualCoords, Point
from bregman.dissimilarity.bregman import BregmanDivergence
from bregman.manifold.manifold import BregmanManifold


class BregmanBall(Ball[BregmanManifold]):
    """Bregman ball defined on Bregman manifolds.

    Parameters:
        bregman_divergence: Bregman divergence being used to calculate the Bregman ball.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        center: Point,
        radius: float,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        """Initialize Bregman ball.

        Args:
            manifold: Bregman manifold which the Bregman ball is defined on.
            center: Ball center.
            radius: Ball radius.
            dcoords: Bregman generator being used to construct Bregman ball.
        """
        super().__init__(manifold, center, radius, dcoords.value)

        self.bregman_divergence = BregmanDivergence(manifold, dcoords=dcoords)

    def is_in(self, other: Point) -> bool:
        """Boolean test if point is in the Bregman ball.

        Args:
            other: Point to be tested.

        Returns:
            Boolean value of if other is in the Bregman ball or not.
        """
        return self.bregman_divergence(self.center, other).item() < self.radius


def bregman_badoiu_clarkson(
    manifold: BregmanManifold,
    points: list[Point],
    T: int,
    dcoords: DualCoords = DualCoords.THETA,
) -> BregmanBall:
    """Generalized Badoiu & Clarkson algorithm for calculating the smallest
    enclosing Bregman ball from a set of points.

    https://link.springer.com/chapter/10.1007/11564096_65

    Args:
        manifold: Bregman manifold which the Bregman ball is defined on.
        points: Points being enclosed by the output Bregman ball.
        T: Number of iterations of the algorithm.
        dcoords: Bregman generator being used to construct Bregman ball.

    Returns:
        Approximate smallest enclosing Bregman ball.
    """
    divergence = manifold.bregman_generator(dcoords).bregman_divergence
    primal_gen = manifold.bregman_generator(dcoords)
    dual_gen = manifold.bregman_generator(dcoords.dual())

    datas = [p.data for p in points]

    c = datas[0]
    d = 0
    for t in range(1, T):
        dist_pairs = [(p, divergence(c, p).item()) for p in datas]
        s, d = min(dist_pairs, key=lambda pair: pair[1])
        c = dual_gen(t / (t + 1) * primal_gen(c) + 1 / (t + 1) * primal_gen(s))

    center = Point(coords=dcoords.value, data=c)
    return BregmanBall(manifold, center, d, dcoords)
