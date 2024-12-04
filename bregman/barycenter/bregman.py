import jax
import jax.numpy as jnp
from jax import Array

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
        coord_avg = jnp.sum(
            jnp.stack([w * t for w, t in zip(nweights, coords_data)]), axis=0
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

        def get_energy(c: Array) -> float:
            weighted_term = sum(
                w * primal_gen(a * c + (1 - a) * t)
                for w, a, t in zip(nweights, alphas, points_data)
            )
            return float(alpha_mid * primal_gen(c) - weighted_term)

        diff = float("inf")
        barycenter = jnp.sum(
            jnp.stack([w * t for w, t in zip(nweights, points_data)]), axis=0
        )
        cur_energy = get_energy(barycenter)
        while diff > eps:
            aw_grads = jnp.stack(
                [
                    a * w * primal_gen.grad(a * barycenter + (1 - a) * t)
                    for w, a, t in zip(nweights, alphas, points_data)
                ]
            )
            avg_grad = jnp.sum(aw_grads, axis=0) / alpha_mid

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


class GaussBregmanCentroid(DualApproxBarycenter):
    r"""Gauss Bregman Centroid on Bregman manifolds.

    https://arxiv.org/pdf/2410.14326
    """

    def barycenter(
        self,
        points: list[Point],
        weights: list[float],
        eps: float = EPS,
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

        assert len(points) == len(weights)

        nweights = jnp.array([w / sum(weights) for w in weights])

        points_data = jnp.array(
            [self.manifold.convert_coord(coord_type, p).data for p in points]
        )

        upper = nweights @ points_data
        vmap_primal_grad = jax.vmap(primal_gen.grad)
        lower = dual_gen.grad(nweights @ vmap_primal_grad(points_data))

        while jnp.linalg.norm(upper - lower) > eps:
            next_upper = 0.5 * (upper + lower)
            next_lower = dual_gen.grad(
                0.5 * (primal_gen.grad(upper) + primal_gen.grad(lower))
            )

            upper, lower = next_upper, next_lower

        # Convert to point
        barycenter_point = Point(coord_type, upper)
        return barycenter_point

    def __call__(
        self,
        points: list[Point],
        weights: list[float] | None = None,
        eps: float = EPS,
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

        return self.barycenter(points, weights, eps=eps)


class JeffreysFisherRaoCentroid(DualBarycenter):
    r"""Jeffreys-Fisher-Rao Centroid on Bregman manifolds.

    https://arxiv.org/pdf/2410.14326
    """

    def barycenter(
        self,
        points: list[Point],
        weights: list[float],
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

        assert len(points) == len(weights)

        nweights = jnp.array([w / sum(weights) for w in weights])
        points_data = jnp.array(
            [self.manifold.convert_coord(coord_type, p).data for p in points]
        )

        vmap_primal_grad = jax.vmap(primal_gen.grad)

        upper_data = nweights @ points_data
        lower_data = dual_gen.grad(nweights @ vmap_primal_grad(points_data))

        upper = Point(coord_type, upper_data)
        lower = Point(coord_type, lower_data)

        fisher_rao_geodesic = self.manifold.fisher_rao_geodesic(upper, lower)
        return fisher_rao_geodesic(0.5)

    def __call__(
        self,
        points: list[Point],
        weights: list[float] | None = None,
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

        return self.barycenter(points, weights)
