from abc import ABC, abstractmethod

import numpy as np

from bregman.object.distribution import Distribution


class ExponentialFamily(Distribution, ABC):
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
