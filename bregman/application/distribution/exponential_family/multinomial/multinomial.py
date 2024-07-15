import math

import autograd.numpy as anp
import numpy as np

from bregman.application.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution, ExponentialFamilyManifold)
from bregman.base import LAMBDA_COORDS, THETA_COORDS, DisplayPoint, Point
from bregman.manifold.generator import AutoDiffGenerator


class MultinomialPoint(DisplayPoint):
    """Display point for the Multinomial manifold."""

    def display(self) -> str:
        """Generated pretty printed string on display.

        Returns:
            String of probability values of Multinomial point.
        """
        return f"Probs: {str(self.data)}"


class MultinomialDistribution(ExponentialFamilyDistribution):
    r"""Multinomial distributions as exponential family distributions.

    Parameters:
        theta: Natural parameters (:math:`p_1, \ldots, p_{k-1}`).
        n: Number of total draws.
    """

    def __init__(self, theta: np.ndarray, n: int) -> None:
        r"""Initialize Multinomial distribution.

        Args:
            theta: Natural parameters (:math:`p_1, \ldots, p_{k-1}`).
            n: Number of total draws.
        """
        super().__init__(theta, (len(theta),))
        self.n = n

    @staticmethod
    def t(x: np.ndarray) -> np.ndarray:
        r""":math:`t(x)` sufficient statistics function of the Multinomial
        distribution.

        Args:
            x: Sample space input.

        Returns:
            Sufficient statistics function of the Multinomial distribution evaluated at x.
        """
        return x

    @staticmethod
    def k(x: np.ndarray) -> np.ndarray:
        r""":math:`k(x)` carrier measure of the Multinomial distribution.

        Args:
            x: Sample space input.

        Returns:
            Carries measure of the Multinomial distribution evaluated at x.
        """
        return -np.sum([np.log(math.factorial(i)) for i in x])

    def F(self, x: np.ndarray) -> np.ndarray:
        r""":math:`F(x) = \log \int \exp(\theta^T t(x)) \mathrm{d}x`
        normalizer of the Multinomial distribution.

        Args:
            x: Parameter value.

        Returns:
            Normalizer of the Multinomial distribution evaluated at parameter value x.
        """
        agg = sum(np.exp(i) for i in x)
        return self.n * np.log(1 + agg) - np.log(math.factorial(self.n))


class MultinomialPrimalGenerator(AutoDiffGenerator):
    """Multinomial manifold primal Bregman generator.

    Parameters:
        n: Number of total draws.
        k: Number of categories.
    """

    def __init__(self, n: int, k: int):
        """Initialize Multinomial manifold primal Bregman generator.

        Args:
            n: Number of total draws.
            k: Number of categories.
        """
        super().__init__(k - 1)

        self.n = n
        self.k = k

    def _F(self, x: np.ndarray) -> np.ndarray:
        """Multinomial manifold primal Bregman generator function.

        Args:
            x: Input value.

        Returns:
            Multinomial manifold primal Bregman generator value evaluated at x.
        """
        agg = anp.exp(x).sum()
        return self.n * anp.log(1 + agg) - anp.log(
            float(math.factorial(self.n))
        )


class MultinomialDualGenerator(AutoDiffGenerator):
    """Multinomial manifold dual Bregman generator.

    Parameters:
        n: Number of total draws.
        k: Number of categories.
    """

    def __init__(self, n: int, k: int):
        """Initialize Multinomial manifold dual Bregman generator.

        Args:
            n: Number of total draws.
            k: Number of categories.
        """
        super().__init__(k - 1)

        self.n = n
        self.k = k

    def _F(self, x: np.ndarray) -> np.ndarray:
        """Multinomial manifold dual Bregman generator function.

        Args:
            x: Input value.

        Returns:
            Multinomial manifold dual Bregman generator value evaluated at x.
        """
        ent = anp.sum(x * anp.log(x))
        other = self.n - anp.sum(x)
        return ent + other * anp.log(other)


class MultinomialManifold(
    ExponentialFamilyManifold[MultinomialPoint, MultinomialDistribution]
):
    """Multinomial exponential family manifold.

    Parameters:
        n: Number of total draws.
        k: Number of categories.
    """

    def __init__(self, k: int, n: int) -> None:
        """Initialize Multinomial exponential family manifold.

        Args:
            n: Number of total draws.
            k: Number of categories.
        """
        self.k = k
        self.n = n

        F_gen = MultinomialPrimalGenerator(n, k)
        G_gen = MultinomialDualGenerator(n, k)

        super().__init__(
            natural_generator=F_gen,
            expected_generator=G_gen,
            distribution_class=MultinomialDistribution,
            display_factory_class=MultinomialPoint,
            dimension=k - 1,
        )

    def point_to_distribution(self, point: Point) -> MultinomialDistribution:
        """Converts a point to a Multinomial distribution.

        Args:
            point: Point to be converted.

        Returns:
            Multinomial distribution corresponding to the point.
        """
        theta = self.convert_coord(THETA_COORDS, point).data

        return MultinomialDistribution(theta, self.n)

    def distribution_to_point(
        self, distribution: MultinomialDistribution
    ) -> MultinomialPoint:
        """Converts a Multinomial distribution to a point in the manifold.

        Args:
            distribution: Multinomial distribution to be converted.

        Returns:
            Point corresponding to the Multinomial distribution.
        """
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
