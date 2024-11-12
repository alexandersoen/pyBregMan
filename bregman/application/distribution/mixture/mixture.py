from abc import ABC
from typing import Generic, Sequence, TypeVar

import jax.numpy as jnp
from jax import Array

from bregman.application.application import LAMBDA_COORDS
from bregman.application.distribution.distribution import DistributionManifold
from bregman.application.distribution.exponential_family.multinomial import (
    MultinomialDualGenerator,
    MultinomialPrimalGenerator,
)
from bregman.base import DisplayPoint, Point
from bregman.dissimilarity.bregman import (
    BregmanDivergence,
    SkewBurbeaRaoDivergence,
)
from bregman.object.distribution import Distribution


class MixingDimensionMissMatch(Exception):
    """Exception when mixing weights do not match the number of distributions
    provided.
    """

    pass


class MixturePoint(DisplayPoint):
    """Display point for Mixture distributions."""

    def display(self) -> str:
        """Generated pretty printed string on display.

        Returns:
            String of weights values of Mixture point.
        """
        return str(self.data)


MixedDistribution = TypeVar("MixedDistribution", bound=Distribution)


class MixtureDistribution(Distribution, Generic[MixedDistribution]):
    """Mixture distributions with mixed mixing components.

    Attributes:
        weights: Mixture weights.
        distributions: Mixture components.
    """

    def __init__(
        self,
        weights: Array,
        distributions: Sequence[MixedDistribution],
    ) -> None:
        """Initialize Mixture distribution.

        Args:
            weights: Mixture weights.
            distributions: Mixture components.
        """
        if len(weights) != len(distributions) - 1:
            raise MixingDimensionMissMatch

        super().__init__(distributions[0].dimension)

        self.weights = weights
        self.distributions = distributions

    def pdf(self, x: Array) -> Array:
        """P.d.f. of a mixture distribution.

        Args:
            x: Sample space input.

        Returns:
            Mixture distribution's p.d.f. evaluated at x.
        """
        all_w = jnp.zeros(len(self.distributions))
        all_w = all_w.at[:-1].set(self.weights)
        all_w = all_w.at[-1].set(1 - jnp.sum(self.weights))

        return jnp.sum(
            jnp.array([w * float(p.pdf(x)) for w, p in zip(all_w, self.distributions)])
        )


class MixtureManifold(
    DistributionManifold[MixturePoint, MixtureDistribution[MixedDistribution]],
    Generic[MixedDistribution],
    ABC,
):
    """Abstract Mixture manifold.

    Attributes:
        distributions: Mixing components.
    """

    def __init__(
        self,
        distributions: Sequence[MixedDistribution],
    ) -> None:
        """Initialize Mixture manifold.

        Args:
            distributions: Mixing components.
        """
        dimension = len(distributions) - 1

        super().__init__(
            MultinomialDualGenerator(1, len(distributions)),
            MultinomialPrimalGenerator(1, len(distributions)),
            MixturePoint,
            dimension,
        )

        self.set_distributions(distributions)

    def set_distributions(self, distributions: Sequence[MixedDistribution]) -> None:
        """Set mixing component distributions of the manifold.

        Args:
            distributions: Mixing components.
        """
        assert len(distributions) == (self.dimension + 1)
        self.distributions = distributions

    def point_to_distribution(self, point: Point) -> MixtureDistribution:
        """Converts a point to a Mixture distribution.

        Args:
            point: Point to be converted.

        Returns:
            Mixture distribution corresponding to the point.
        """
        weights = self.convert_coord(LAMBDA_COORDS, point).data

        return MixtureDistribution(weights, self.distributions)

    def distribution_to_point(self, distribution: MixtureDistribution) -> MixturePoint:
        """Converts a Mixture distribution to a point in the manifold.

        Args:
            distribution: Mixture distribution to be converted.

        Returns:
            Point corresponding to the Mixture distribution.
        """
        return self.convert_to_display(
            Point(
                coords=LAMBDA_COORDS,
                data=distribution.weights,
            )
        )

    def kl_divergence(self, point_1: Point, point_2: Point) -> Array:
        """KL-Divergence of two points in a Mixture manifold.

        Args:
            point_1: Left-sided argument of the KL-Divergence.
            point_2: Right-sided argument of the KL-Divergence.

        Returns:
            KL-Divergence between point_1 and point_2 on the Mixture manifold.
        """
        breg_div = BregmanDivergence(self)
        return breg_div(point_1, point_2)

    def jensen_shannon_divergence(self, point_1: Point, point_2: Point) -> Array:
        """Jensen-Shannon Divergence of two points in a Mixture manifold.

        Args:
            point_1: Left-sided argument of the Jensen-Shannon Divergence.
            point_2: Right-sided argument of the Jensen-Shannon Divergence.

        Returns:
            Jensen-Shannon Divergence between point_1 and point_2 on the Mixture manifold.
        """
        br_div = SkewBurbeaRaoDivergence(self, alpha=0.5)
        return br_div(point_1, point_2)

    def _lambda_to_theta(self, lamb: Array) -> Array:
        return lamb

    def _lambda_to_eta(self, lamb: Array) -> Array:
        return self._theta_to_eta(lamb)

    def _theta_to_lambda(self, theta: Array) -> Array:
        return theta

    def _eta_to_lambda(self, eta: Array) -> Array:
        return self._eta_to_theta(eta)
