from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.application.application import MyDisplayPoint
from bregman.application.distribution.distribution import DistributionManifold
from bregman.base import Point
from bregman.dissimilarity.bregman import BregmanDivergence
from bregman.manifold.generator import Generator
from bregman.object.distribution import Distribution


class ExponentialFamilyDistribution(Distribution, ABC):
    """Currently assuming base measure is unit."""

    theta: np.ndarray  # theta parameters

    @staticmethod
    @abstractmethod
    def t(x: np.ndarray) -> np.ndarray:
        r"""t(x) sufficient statistics function."""
        pass

    @staticmethod
    @abstractmethod
    def k(x: np.ndarray) -> np.ndarray:
        r"""k(x) carrier measure."""
        pass

    @abstractmethod
    def F(self, x: np.ndarray) -> np.ndarray:
        r"""F(x) = \log \int \exp(\theta^\T t(x)) dx normalizer"""
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        inner = np.dot(self.theta, self.t(x))
        return np.exp(inner - self.F(self.theta) + self.k(x))


MyExpFamDistribution = TypeVar(
    "MyExpFamDistribution", bound=ExponentialFamilyDistribution
)


class ExponentialFamilyManifold(
    DistributionManifold[MyDisplayPoint, MyExpFamDistribution],
    Generic[MyDisplayPoint, MyExpFamDistribution],
    ABC,
):
    def __init__(
        self,
        natural_generator: Generator,
        expected_generator: Generator,
        distribution_class: type[MyExpFamDistribution],
        display_factory_class: type[MyDisplayPoint],
        dimension: int,
    ) -> None:
        super().__init__(
            natural_generator,
            expected_generator,
            display_factory_class,
            dimension,
        )

        self.eta_generator = expected_generator  # Fix typing
        self.distribution_class = distribution_class

    def t(self, x: np.ndarray) -> np.ndarray:
        return self.distribution_class.t(x)

    def kl_divergence(self, point_1: Point, point_2: Point) -> np.ndarray:
        breg_div = BregmanDivergence(self)
        return breg_div(point_1, point_2)
