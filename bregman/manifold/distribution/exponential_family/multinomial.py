import math

import autograd.numpy as anp
import numpy as np

from bregman.base import DisplayPoint, Point
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import LAMBDA_COORDS, point_convert_wrapper
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.manifold.distribution.exponential_family.exp_family import \
    ExponentialFamily
from bregman.manifold.manifold import THETA_COORDS


class MultinomialPoint(DisplayPoint):

    def __init__(self, coords, data) -> None:
        super().__init__(coords, data)

    def display(self) -> str:
        return str(self.data)


class MultinomialDistribution(ExponentialFamily):

    def __init__(self, theta: np.ndarray, n: int) -> None:
        self.theta = theta
        self.n = n

    def t(self, x: np.ndarray) -> np.ndarray:
        return x

    def k(self, x: np.ndarray) -> np.ndarray:
        return -np.sum([np.log(math.factorial(i)) for i in x])

    def F(self, x: np.ndarray) -> np.ndarray:
        agg = sum(np.exp(i) for i in x)
        return self.n * np.log(1 + agg) - np.log(math.factorial(self.n))


class MultinomialPrimalGenerator(AutoDiffGenerator):

    def __init__(self, n: int, k: int):
        super().__init__()

        self.n = n
        self.k = k

        self.dimension = k - 1

    def _F(self, x: np.ndarray) -> np.ndarray:
        agg = anp.exp(x).sum()
        return self.n * anp.log(1 + agg) - anp.log(
            float(math.factorial(self.n))
        )


class MultinomialDualGenerator(AutoDiffGenerator):

    def __init__(self, n: int, k: int):
        super().__init__()

        self.n = n
        self.k = k

        self.dimension = k - 1

    def _F(self, x: np.ndarray) -> np.ndarray:
        ent = anp.sum(x * anp.log(x))
        other = self.n - anp.sum(x)
        return ent + other * anp.log(other)


class MultinomialManifold(
    DistributionManifold[MultinomialPoint, MultinomialDistribution]
):

    def __init__(
        self,
        k: int,
        n: int,
    ) -> None:

        self.k = k
        self.n = n

        F_gen = MultinomialPrimalGenerator(n, k)
        G_gen = MultinomialDualGenerator(n, k)

        super().__init__(
            F_gen,
            G_gen,
            point_convert_wrapper(MultinomialPoint),
            dimension=k - 1,
        )

    def point_to_distribution(self, point: Point) -> MultinomialDistribution:
        theta = self.convert_coord(THETA_COORDS, point).data

        return MultinomialDistribution(theta, self.n)

    def distribution_to_point(
        self, distribution: MultinomialDistribution
    ) -> MultinomialPoint:
        return self.convert_to_display(
            Point(
                coords=LAMBDA_COORDS,
                data=distribution.theta,
            )
        )

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        return np.log(lamb[:-1] / lamb[-1])

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        return self.n * lamb[:-1]

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        norm = 1 + np.exp(theta).sum()

        lamb = np.zeros(self.k)
        lamb[:-1] = np.exp(theta) / norm
        lamb[-1] = 1 / norm
        return lamb

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        lamb = np.zeros(self.k)
        lamb[:-1] = eta / self.n
        lamb[-1] = (self.n - eta.sum()) / self.n
        return lamb