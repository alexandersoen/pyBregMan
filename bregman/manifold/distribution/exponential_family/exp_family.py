from abc import ABC, abstractmethod

import numpy as np

from bregman.object.distribution import Distribution


class ExponentialFamily(Distribution, ABC):
    """Currently assuming base measure is unit."""

    theta: np.ndarray  # theta parameters

    @abstractmethod
    def sufficient_statistic(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def normalizer(self, x: np.ndarray) -> np.ndarray:
        r"""F(x) = \log \int \exp(\theta^\T t(x)) dx"""
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        inner = np.dot(self.theta, self.sufficient_statistic(x))
        return np.exp(inner - self.normalizer)
