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

    def bergman_divergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.F(x) - self.F(y) - np.inner(self.grad(y), x - y)

    def skew_jensen_bregman_divergence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alphas: list[float],
        weights: list[float],
    ) -> np.ndarray:
        alpha_mid = sum(w * a for w, a in zip(weights, alphas))

        xy_alpha_list = [(1 - a) * x + a * y for a in alphas]
        xy_alpha_mid = (1 - alpha_mid) * x + alpha_mid * y

        bregs = np.stack(
            [
                w * self.bergman_divergence(xy, xy_alpha_mid)
                for w, xy in zip(weights, xy_alpha_list)
            ]
        )
        return np.sum(bregs, axis=0)

    def skew_burbea_rao_divergence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        const = 1 / (alpha * (1 - alpha))
        xy_alpha = alpha * x + (1 - alpha) * y
        return const * (
            alpha * self.F(x) + (1 - alpha) * self.F(y) - self.F(xy_alpha)
        )

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
