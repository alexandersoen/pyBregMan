import numpy as np

from bregman.base import Point
from bregman.generator.generator import Generator
from bregman.manifold.application import ORDINARY_COORDS
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.object.distribution import Distribution


class MixingDimensionMissMatch(Exception):
    pass


class MixtureDistribution(Distribution):

    def __init__(
        self,
        mixture_w: np.ndarray,
        base_distribution: Distribution,
        other_distributions: list[Distribution],
    ) -> None:
        super().__init__()

        if len(mixture_w) != len(other_distributions):
            raise MixingDimensionMissMatch

        self.mixture_w = mixture_w
        self.base_distribution = base_distribution
        self.other_distributions = other_distributions

    def pdf(self, x: np.ndarray) -> np.ndarray:
        base_w = 1 - np.sum(self.mixture_w)

        base_pdf = base_w * self.base_distribution.pdf(x)
        other_pdf = sum(
            w * p.pdf(x)
            for p, w in zip(self.mixture_w, self.other_distributions)
        )

        return base_pdf + other_pdf


class MixtureManifold(DistributionManifold[MixtureDistribution]):

    def __init__(
        self,
        base_distribution: Distribution,
        other_distributions: list[Distribution],
        natural_generator: Generator,
        expected_generator: Generator,
    ) -> None:
        dimension = len(other_distributions)

        super().__init__(natural_generator, expected_generator, dimension)

        self.base_distribution = base_distribution
        self.other_distributions = other_distributions

    def point_to_distribution(self, point: Point) -> MixtureDistribution:
        mixing_w = self.convert_coord(ORDINARY_COORDS, point).data

        return MixtureDistribution(
            mixing_w, self.base_distribution, self.other_distributions
        )

    def distribution_to_point(
        self, distribution: MixtureDistribution
    ) -> Point:
        return Point(
            coords=ORDINARY_COORDS,
            data=distribution.mixture_w,
        )

    def _ordinary_to_natural(self, lamb: np.ndarray) -> np.ndarray:
        return lamb

    def _ordinary_to_moment(self, lamb: np.ndarray) -> np.ndarray:
        return self._natural_to_moment(lamb)

    def _natural_to_ordinary(self, theta: np.ndarray) -> np.ndarray:
        return theta

    def _moment_to_ordinary(self, eta: np.ndarray) -> np.ndarray:
        return self._moment_to_natural(eta)
