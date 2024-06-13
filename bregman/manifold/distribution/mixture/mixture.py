from abc import ABC
from typing import Sequence

import numpy as np

from bregman.base import DisplayPoint, Point
from bregman.generator.generator import Generator
from bregman.manifold.application import LAMBDA_COORDS
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.object.distribution import Distribution


class MixingDimensionMissMatch(Exception):
    pass


class MixturePoint(DisplayPoint):

    def display(self) -> str:
        return str(self.data)


class MixtureDistribution(Distribution):

    def __init__(
        self,
        weights: np.ndarray,
        distributions: Sequence[Distribution],
    ) -> None:
        super().__init__()

        if len(weights) != len(distributions) - 1:
            raise MixingDimensionMissMatch

        self.weights = weights
        self.distributions = distributions

    def pdf(self, x: np.ndarray) -> np.ndarray:
        all_w = np.zeros(len(self.distributions))
        all_w[:-1] = self.weights
        all_w[-1] = 1 - np.sum(self.weights)

        return np.sum(
            [w * float(p.pdf(x)) for w, p in zip(all_w, self.distributions)]
        )


class MixtureManifold(
    DistributionManifold[MixturePoint, MixtureDistribution], ABC
):

    def __init__(
        self,
        distributions: Sequence[Distribution],
        natural_generator: Generator,
        expected_generator: Generator,
    ) -> None:
        dimension = len(distributions) - 1

        super().__init__(
            natural_generator,
            expected_generator,
            MixturePoint,
            dimension,
        )

        self.distributions = distributions

    def point_to_distribution(self, point: Point) -> MixtureDistribution:
        weights = self.convert_coord(LAMBDA_COORDS, point).data

        return MixtureDistribution(weights, self.distributions)

    def distribution_to_point(
        self, distribution: MixtureDistribution
    ) -> MixturePoint:
        return self.convert_to_display(
            Point(
                coords=LAMBDA_COORDS,
                data=distribution.weights,
            )
        )

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        return lamb

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        return self._theta_to_eta(lamb)

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        return theta

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        return self._eta_to_theta(eta)
