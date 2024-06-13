import autograd.numpy as anp
import numpy as np

from bregman.base import DisplayPoint
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import ApplicationManifold


class NotPSDMatrix(Exception):
    pass


class PSDPoint(DisplayPoint):

    @property
    def dimension(self) -> int:
        return int(0.5 * (np.sqrt(8 * len(self.data) + 1) - 1))

    @property
    def m(self) -> np.ndarray:
        return psd_data_to_matrices(self.data, self.dimension)

    def display(self) -> str:
        return str(self.m)


class PSDPrimalGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

        triu_indices = np.triu_indices(self.dimension)
        selector = np.arange(self.dimension * self.dimension).reshape(
            self.dimension, self.dimension
        )[triu_indices]

        self.selector = selector

    def _pre_autodiff(self, x: np.ndarray) -> np.ndarray:
        return psd_data_to_matrices(x, self.dimension).flatten()

    def _post_grad(self, x: np.ndarray) -> np.ndarray:
        return x[self.selector]

    def _post_hess(self, x: np.ndarray) -> np.ndarray:
        return x[np.ix_(self.selector, self.selector)]

    def _F(self, x: np.ndarray) -> np.ndarray:
        m = x.reshape(self.dimension, self.dimension)
        return -anp.log(anp.linalg.det(m))


class PSDDualGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

        triu_indices = np.triu_indices(self.dimension)
        selector = np.arange(self.dimension * self.dimension).reshape(
            self.dimension, self.dimension
        )[triu_indices]

        self.selector = selector

    def _pre_autodiff(self, x: np.ndarray) -> np.ndarray:
        return psd_data_to_matrices(x, self.dimension).flatten()

    def _post_grad(self, x: np.ndarray) -> np.ndarray:
        return x[self.selector]

    def _post_hess(self, x: np.ndarray) -> np.ndarray:
        return x[np.ix_(self.selector, self.selector)]

    def _F(self, x: np.ndarray) -> np.ndarray:
        m = x.reshape(self.dimension, self.dimension)
        return anp.log(anp.linalg.det(anp.linalg.inv(m))) - self.dimension


class PSDManifold(ApplicationManifold[PSDPoint]):

    def __init__(self, n_dimension: int):
        F_gen = PSDPrimalGenerator(n_dimension)
        G_gen = PSDDualGenerator(n_dimension)

        super().__init__(
            natural_generator=F_gen,
            expected_generator=G_gen,
            display_factory_class=PSDPoint,
            dimension=int(n_dimension * (n_dimension + 1) / 2),
        )

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        return lamb

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        return self._theta_to_eta(lamb)

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        return theta

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        return self._eta_to_theta(eta)


def psd_data_from_matrices(m: np.ndarray, dimension: int) -> np.ndarray:
    return m[np.triu_indices(dimension)]


def psd_data_to_matrices(d: np.ndarray, dimension: int) -> np.ndarray:
    m = np.empty((dimension, dimension))
    m[np.triu_indices(dimension)] = d
    m[np.tril_indices(dimension)] = d

    return m
