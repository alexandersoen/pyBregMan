from abc import ABC, abstractmethod

import autograd
import numpy as np


class Generator(ABC):

    dimension: int

    @abstractmethod
    def F(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hess(self, x: np.ndarray) -> np.ndarray:
        pass

    def divergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.F(x) - self.F(y) - np.inner(self.grad(y), x - y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.F(x)


class AutoDiffGenerator(Generator, ABC):

    def _pre_autodiff(self, x: np.ndarray) -> np.ndarray:
        return x

    @abstractmethod
    def _F(self, x: np.ndarray) -> np.ndarray:
        pass

    def F(self, x: np.ndarray) -> np.ndarray:
        y = self._pre_autodiff(x)
        return self._F(y)

    def _post_grad(self, x: np.ndarray) -> np.ndarray:
        return x

    def grad(self, x: np.ndarray) -> np.ndarray:
        y = self._pre_autodiff(x)
        z = autograd.grad(self._F)(y)
        return self._post_grad(z)

    def _post_hess(self, x: np.ndarray) -> np.ndarray:
        return x

    def hess(self, x: np.ndarray) -> np.ndarray:
        y = self._pre_autodiff(x)
        z = autograd.hessian(self._F)(y)
        return self._post_hess(z)
