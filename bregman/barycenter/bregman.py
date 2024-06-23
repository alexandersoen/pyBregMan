import numpy as np

from bregman.barycenter.base import ApproxBarycenter, Barycenter
from bregman.base import Point
from bregman.constants import EPS
from bregman.manifold.manifold import BregmanManifold, DualCoord


class DualBarycenter(Barycenter[BregmanManifold]):

    def __init__(
        self, manifold: BregmanManifold, coord: DualCoord = DualCoord.THETA
    ) -> None:
        super().__init__(manifold)

        self.coord = coord


class DualApproxBarycenter(ApproxBarycenter[BregmanManifold]):

    def __init__(
        self,
        manifold: BregmanManifold,
        coord: DualCoord = DualCoord.THETA,
    ) -> None:
        super().__init__(manifold)

        self.coord = coord


class BregmanBarycenter(DualBarycenter):

    def barycenter(self, points: list[Point], weights: list[float]) -> Point:

        nweights = [w / sum(weights) for w in weights]

        coords_data = [
            self.manifold.convert_coord(self.coord.value, p).data
            for p in points
        ]
        coord_avg = np.sum(
            np.stack([w * t for w, t in zip(nweights, coords_data)]), axis=0
        )
        return Point(self.coord.value, coord_avg)


class SkewBurbeaRaoBarycenter(DualApproxBarycenter):

    def __init__(
        self,
        manifold: BregmanManifold,
        coord: DualCoord = DualCoord.THETA,
    ) -> None:
        super().__init__(manifold, coord)

    def barycenter(
        self,
        points: list[Point],
        weights: list[float],
        eps: float = EPS,
        alphas: list[float] | None = None,
    ) -> Point:
        """
        https://arxiv.org/pdf/1004.5049
        """

        coord_type = self.coord.value
        primal_gen = self.manifold.bregman_generator(self.coord)
        dual_gen = self.manifold.bregman_generator(self.coord.dual())

        if alphas is None:
            alphas = [0.5] * len(weights)

        assert len(points) == len(alphas) == len(weights)

        nweights = [w / sum(weights) for w in weights]
        alpha_mid = sum(w * a for w, a in zip(nweights, alphas))
        points_data = [
            self.manifold.convert_coord(coord_type, p).data for p in points
        ]

        def get_energy(p: np.ndarray) -> float:
            weighted_term = sum(
                w * primal_gen(a * p + (1 - a) * t)
                for w, a, t in zip(nweights, alphas, points_data)
            )
            return float(alpha_mid * primal_gen(p) - weighted_term)

        diff = float("inf")
        barycenter = np.sum(
            np.stack([w * t for w, t in zip(nweights, points_data)]), axis=0
        )
        cur_energy = get_energy(barycenter)
        while diff > eps:
            aw_grads = np.stack(
                [
                    a * w * primal_gen.grad(a * barycenter + (1 - a) * t)
                    for w, a, t in zip(nweights, alphas, points_data)
                ]
            )
            avg_grad = np.sum(aw_grads, axis=0)

            # Update
            barycenter = dual_gen.grad(avg_grad / alpha_mid)

            new_energy = get_energy(barycenter)
            diff = abs(new_energy - cur_energy)
            cur_energy = new_energy

        # Convert to point
        barycenter_point = Point(coord_type, barycenter)
        return barycenter_point

    def __call__(
        self,
        points: list[Point],
        weights: list[float],
        eps: float = EPS,
        alphas: list[float] | None = None,
    ) -> Point:
        return self.barycenter(points, weights, eps=eps, alphas=alphas)
