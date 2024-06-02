from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.base import Coordinates, Point
from bregman.manifold.application import MyDisplayPoint
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS
from bregman.object.distribution import Distribution


class ExponentialFamilyDistribution(Distribution, ABC):
    """Currently assuming base measure is unit."""

    theta: np.ndarray  # theta parameters

    @abstractmethod
    def t(self, x: np.ndarray) -> np.ndarray:
        r"""t(x) sufficient statistics function."""
        pass

    @abstractmethod
    def k(self, x: np.ndarray) -> np.ndarray:
        r"""k(x) carrier measure."""
        pass

    @abstractmethod
    def F(self, x: np.ndarray) -> np.ndarray:
        r"""F(x) = \log \int \exp(\theta^\T t(x)) dx normalizer"""
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        inner = np.dot(self.theta, self.t(x))
        return np.exp(inner - self.F(x) + self.k(x))


MyExpFamDistribution = TypeVar(
    "MyExpFamDistribution", bound=ExponentialFamilyDistribution
)


class ExponentialFamilyManifold(
    DistributionManifold[MyDisplayPoint, MyExpFamDistribution],
    Generic[MyDisplayPoint, MyExpFamDistribution],
    ABC,
):
    def theta_barycenter(
        self, points: list[Point], weights: list[float]
    ) -> Point:
        assert len(points) == len(weights)

        nweights = [w / sum(weights) for w in weights]
        thetas_data = [
            self.convert_coord(THETA_COORDS, p).data for p in points
        ]
        theta_avg = np.sum(
            np.stack([w * t for w, t in zip(nweights, thetas_data)]), axis=0
        )
        return Point(THETA_COORDS, theta_avg)

    def eta_barycenter(
        self, points: list[Point], weights: list[float]
    ) -> Point:
        assert len(points) == len(weights)

        nweights = [w / sum(weights) for w in weights]
        etas_data = [self.convert_coord(ETA_COORDS, p).data for p in points]
        eta_avg = np.sum(
            np.stack([w * t for w, t in zip(nweights, etas_data)]), axis=0
        )
        return Point(ETA_COORDS, eta_avg)

    def theta_skew_burbea_rao_barycenter(
        self,
        points: list[Point],
        alphas: list[float],
        weights: list[float],
        eps: float = 1e-8,
    ) -> Point:
        return self._skew_burbea_rao_barycenter(
            points, alphas, weights, THETA_COORDS, eps
        )

    def eta_skew_burbea_rao_barycenter(
        self,
        points: list[Point],
        alphas: list[float],
        weights: list[float],
        eps: float = 1e-8,
    ) -> Point:
        return self._skew_burbea_rao_barycenter(
            points, alphas, weights, ETA_COORDS, eps
        )

    def bhattacharyya_distance(
        self, point_1: Point, point_2: Point, alpha: float
    ) -> np.ndarray:
        theta_1 = self.convert_coord(THETA_COORDS, point_1)
        theta_2 = self.convert_coord(THETA_COORDS, point_2)

        geodesic = self.theta_geodesic(theta_1, theta_2)
        theta_alpha = geodesic(alpha)

        F_1 = self.theta_generator(theta_1.data)
        F_2 = self.theta_generator(theta_2.data)
        F_alpha = self.theta_generator(theta_alpha.data)

        return alpha * F_1 + (1 - alpha) * F_2 - F_alpha

    def chernoff_point(
        self, point_1: Point, point_2: Point, eps: float = 1e-8
    ) -> float:
        theta_1 = self.convert_coord(THETA_COORDS, point_1)
        theta_2 = self.convert_coord(THETA_COORDS, point_2)

        geodesic = self.theta_geodesic(theta_1, theta_2)

        alpha_min, alpha_mid, alpha_max = 0.0, 0.5, 1.0
        while abs(alpha_max - alpha_min) > eps:
            alpha_mid = 0.5 * (alpha_min + alpha_max)

            theta_alpha = geodesic(alpha_mid)

            bd_1 = self.theta_divergence(theta_1, theta_alpha)
            bd_2 = self.theta_divergence(theta_2, theta_alpha)
            if bd_1 < bd_2:
                alpha_min = alpha_mid
            else:
                alpha_max = alpha_mid

        return 1 - 0.5 * (alpha_min + alpha_max)

    def chernoff_information(self, point_1: Point, point_2: Point):
        alpha_star = self.chernoff_point(point_1, point_2)
        return self.bhattacharyya_distance(point_1, point_2, alpha_star)

    def _skew_burbea_rao_barycenter(
        self,
        points: list[Point],
        alphas: list[float],
        weights: list[float],
        gen_type: Coordinates,
        eps: float = 1e-8,
    ) -> Point:
        """
        https://arxiv.org/pdf/1004.5049
        """
        if gen_type == THETA_COORDS:
            coord_type = THETA_COORDS
            primal_gen = self.theta_generator
            dual_gen = self.eta_generator

        elif gen_type == ETA_COORDS:
            coord_type = ETA_COORDS
            primal_gen = self.eta_generator
            dual_gen = self.theta_generator

        else:
            raise ValueError()

        assert len(points) == len(alphas) == len(weights)

        nweights = [w / sum(weights) for w in weights]
        alpha_mid = sum(w * a for w, a in zip(nweights, alphas))
        points_data = [self.convert_coord(coord_type, p).data for p in points]

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
