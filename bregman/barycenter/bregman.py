import numpy as np

from bregman.barycenter.base import ApproxBarycenter, Barycenter
from bregman.base import DualCoords, Point
from bregman.constants import EPS
from bregman.manifold.manifold import BregmanManifold


class DualBarycenter(Barycenter[BregmanManifold]):
    """Barycenter based on the dual coordinates of Bregman manifolds.

    Parameters:
        coord: Dual coordinates for the barycenter.
    """

    def __init__(
        self, manifold: BregmanManifold, dcoords: DualCoords = DualCoords.THETA
    ) -> None:
        """Initialize barycenter on dual coordinates.

        Args:
            manifold: Bregman manifold which the barycenter is defined on.
            dcoords: Coordinates in which the barycenter is defined on.
        """
        super().__init__(manifold)

        self.coord = dcoords


class DualApproxBarycenter(ApproxBarycenter[BregmanManifold]):
    """Approximate barycenter based on the dual coordinates of Bregman
    manifolds.

    Parameters:
        coord: Dual coordinates for the barycenter.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        """Initialize approximate barycenter on dual coordinates.

        Args:
            manifold: Bregman manifold which the barycenter is defined on.
            dcoords: Coordinates in which the barycenter is defined on.
        """
        super().__init__(manifold)

        self.coord = dcoords


class BregmanBarycenter(DualBarycenter):
    """Bregman barycenter on a Bregman manifold."""

    def barycenter(self, points: list[Point], weights: list[float]) -> Point:
        r"""Bregman barycenter of points with weights.

        .. math:: \min_{c} \sum_{i=1}^{n} B_F(p_i : c).

        This corresponds to taking a weighted average on points in the
        appropriate dual coordinates.

        Args:
            points: Points which the Bregman barycenter is being calculated for.
            weights: Weights for each of the points in the Bregman barycenter.

        Returns:
            Bregman barycenter of points with weights.
        """
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
    r"""Skew Burea-Rao Barycenter on Bregman manifolds.

    https://arxiv.org/pdf/1004.5049
    """

    def barycenter(
        self,
        points: list[Point],
        weights: list[float],
        eps: float = EPS,
        alphas: list[float] | None = None,
    ) -> Point:
        r"""Calculates the skew Burea-Rao barycenter over a vector of skew
        parameters. This is equivalent to calculating the barycenter over a
        list of different (Burea-Rao-type) divergences.

        The barycenter is equivalent to the minimization:

        .. math:: \min_c \left( \sum_{i=1}^n w_i \alpha_i \right) F(c) - \left( \sum_{i=1}^n w_i F(\alpha_i \cdot c + (1-\alpha_i) \cdot p_i) \right).

        This can be approximately solved via a ConCave-Convex Procedure (CCCP).
        See: https://arxiv.org/pdf/1004.5049

        Args:
            points: Points which the skew Burbea-Rao barycenter is being calculated for.
            weights: Weights for each of the points in the skew Burbea-Rao barycenter.
            eps: CCCP iteration progress tolerance.
            alphas: Burbea-Rao :math:`\alpha` skew vector.

        Returns:
            Approximate skew Burea-Rao barycenter calculated using CCCP with eps tolerance.
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

        def get_energy(c: np.ndarray) -> float:
            weighted_term = sum(
                w * primal_gen(a * c + (1 - a) * t)
                for w, a, t in zip(nweights, alphas, points_data)
            )
            return float(alpha_mid * primal_gen(c) - weighted_term)

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
            avg_grad = np.sum(aw_grads, axis=0) / alpha_mid

            # Update
            barycenter = dual_gen.grad(avg_grad)

            new_energy = get_energy(barycenter)
            diff = abs(new_energy - cur_energy)
            cur_energy = new_energy

        # Convert to point
        barycenter_point = Point(coord_type, barycenter)
        return barycenter_point

    def __call__(
        self,
        points: list[Point],
        weights: list[float] | None = None,
        eps: float = EPS,
        alphas: list[float] | None = None,
    ) -> Point:
        r"""Calculates the skew Burea-Rao barycenter over a vector of skew
        parameters. This is equivalent to calculating the barycenter over a
        list of different (Burea-Rao-type) divergences.

        The barycenter is equivalent to the minimization:

        .. math:: \min_c \left( \sum_{i=1}^n w_i \alpha_i \right) F(c) - \left( \sum_{i=1}^n w_i F(\alpha_i \cdot c + (1-\alpha_i) \cdot p_i) \right).

        This can be approximately solved via a ConCave-Convex Procedure (CCCP).
        See: https://arxiv.org/pdf/1004.5049

        Args:
            points: Points which the skew Burbea-Rao barycenter is being calculated for.
            weights: Weights for each of the points in the skew Burbea-Rao barycenter.
            eps: CCCP iteration progress tolerance.
            alphas: Burbea-Rao :math:`\alpha` skew vector.

        Returns:
            Approximate skew Burea-Rao barycenter calculated using CCCP with eps tolerance.
        """
        if weights is None:
            weights = [1.0] * len(points)

        return self.barycenter(points, weights, eps=eps, alphas=alphas)
