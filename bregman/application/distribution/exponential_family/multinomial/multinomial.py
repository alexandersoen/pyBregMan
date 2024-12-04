import math

import jax.numpy as jnp
from jax import Array

from bregman.application.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution,
    ExponentialFamilyManifold,
)
from bregman.application.distribution.exponential_family.multinomial.geodesic import (
    FisherRaoMultinomialGeodesic,
)
from bregman.base import LAMBDA_COORDS, THETA_COORDS, DisplayPoint, Point
from bregman.manifold.generator import AutoDiffGenerator
from bregman.manifold.geodesic import Geodesic


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

    def __init__(self, theta: Array, n: int) -> None:
        r"""Initialize Multinomial distribution.

        Args:
            theta: Natural parameters (:math:`p_1, \ldots, p_{k-1}`).
            n: Number of total draws.
        """
        super().__init__(theta, (len(theta),))
        self.n = n

    @staticmethod
    def t(x: Array) -> Array:
        r""":math:`t(x)` sufficient statistics function of the Multinomial
        distribution.

        Args:
            x: Sample space input.

        Returns:
            Sufficient statistics function of the Multinomial distribution evaluated at x.
        """
        return x

    @staticmethod
    def k(x: Array) -> Array:
        r""":math:`k(x)` carrier measure of the Multinomial distribution.

        Args:
            x: Sample space input.

        Returns:
            Carries measure of the Multinomial distribution evaluated at x.
        """
        return -jnp.sum(jnp.array([jnp.log(math.factorial(i)) for i in x]))

    def F(self, x: Array) -> Array:
        r""":math:`F(x) = \log \int \exp(\theta^T t(x)) \mathrm{d}x`
        normalizer of the Multinomial distribution.

        Args:
            x: Parameter value.

        Returns:
            Normalizer of the Multinomial distribution evaluated at parameter value x.
        """
        agg = sum(jnp.exp(i) for i in x)
        return self.n * jnp.log(1 + agg) - jnp.log(math.factorial(self.n))


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

    def _F(self, x: Array) -> Array:
        """Multinomial manifold primal Bregman generator function.

        Args:
            x: Input value.

        Returns:
            Multinomial manifold primal Bregman generator value evaluated at x.
        """
        agg = jnp.exp(x).sum()
        return self.n * jnp.log(1 + agg) - jnp.log(
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

    def _F(self, x: Array) -> Array:
        """Multinomial manifold dual Bregman generator function.

        Args:
            x: Input value.

        Returns:
            Multinomial manifold dual Bregman generator value evaluated at x.
        """
        ent = jnp.sum(x * jnp.log(x))
        other = self.n - jnp.sum(x)
        return ent + other * jnp.log(other)


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
        self.n = n
        self.k = k

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

    def _lambda_to_theta(self, lamb: Array) -> Array:
        return jnp.log(lamb[:-1] / lamb[-1])

    def _lambda_to_eta(self, lamb: Array) -> Array:
        return self.n * lamb[:-1]

    def _theta_to_lambda(self, theta: Array) -> Array:
        norm = 1 + jnp.exp(theta).sum()

        lamb = jnp.zeros(self.k)
        lamb = lamb.at[:-1].set(jnp.exp(theta) / norm)
        lamb = lamb.at[-1].set(1 / norm)
        return lamb

    def _eta_to_lambda(self, eta: Array) -> Array:
        lamb = jnp.zeros(self.k)
        lamb = lamb.at[:-1].set(eta / self.n)
        lamb = lamb.at[-1].set((self.n - eta.sum()) / self.n)
        return lamb

    def fisher_rao_geodesic(self, source: Point, dest: Point) -> Geodesic:
        return FisherRaoMultinomialGeodesic(self, source, dest)
