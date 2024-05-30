from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.base import Point
from bregman.manifold.application import MyDisplayPoint
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.manifold.manifold import THETA_COORDS
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
