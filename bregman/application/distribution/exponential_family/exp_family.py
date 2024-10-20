from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike

from bregman.application.application import MyDisplayPoint
from bregman.application.distribution.distribution import DistributionManifold
from bregman.base import Point, Shape
from bregman.dissimilarity.bregman import BregmanDivergence
from bregman.manifold.generator import Generator
from bregman.object.distribution import Distribution


class ExponentialFamilyDistribution(Distribution, ABC):
    """Exponential family distribution abstract class.

    Parameters:
        theta: Natural parameters of the exponential family distribution.
    """

    def __init__(self, theta: ArrayLike, dimension: Shape) -> None:
        """Initialize exponential family distribution.

        Args:
            theta: Natural parameter of distribution.
            dimension: Shape of natural parameters.
        """
        super().__init__(dimension)
        self.theta = theta

    @staticmethod
    @abstractmethod
    def t(x: ArrayLike) -> Array:
        r""":math:`t(x)` sufficient statistics function.

        Args:
            x: Sample space input.

        Returns:
            Sufficient statistics function evaluated at x.
        """
        pass

    @staticmethod
    @abstractmethod
    def k(x: ArrayLike) -> Array:
        r""":math:`k(x)` carrier measure.

        Args:
            x: Sample space input.

        Returns:
            Carries measure evaluated at x.
        """
        pass

    @abstractmethod
    def F(self, x: ArrayLike) -> Array:
        r""":math:`F(x) = \log \int \exp(\theta^T t(x)) \mathrm{d}x` normalizer.

        Args:
            x: Parameter value.

        Returns:
            Normalizer evaluated at parameter value x.
        """
        pass

    def pdf(self, x: ArrayLike) -> Array:
        """P.d.f. of exponential family distribution.

        Args:
            x: Sample space input.

        Returns:
            P.d.f. of exponential family distribution evaluated at x.
        """
        inner = jnp.dot(self.theta, self.t(x))
        return jnp.exp(inner - self.F(self.theta) + self.k(x))


MyExpFamDistribution = TypeVar(
    "MyExpFamDistribution", bound=ExponentialFamilyDistribution
)


class ExponentialFamilyManifold(
    DistributionManifold[MyDisplayPoint, MyExpFamDistribution],
    Generic[MyDisplayPoint, MyExpFamDistribution],
    ABC,
):
    """Exponential family distribution manifold.

    Parameters:
        distribution_class: Distribution class corresponding to the manifold.
    """

    def __init__(
        self,
        natural_generator: Generator,
        expected_generator: Generator,
        distribution_class: type[MyExpFamDistribution],
        display_factory_class: type[MyDisplayPoint],
        dimension: int,
    ) -> None:
        r"""Initialize exponential family distribution manifold.

        Args:
            natural_generator: Primal generator for :math:`\theta`-coordinates.
            expected_generator: Dual generator for :math:`\eta`-coordinates. Optional.
            distribution_class: Distribution class corresponding to the manifold.
            display_factory_class: Constructor for display point of the distribution manifold.
            dimension: Dimension of canonical parameterizations (:math:`\theta`-or :math:`\eta`-coordinates).
        """
        super().__init__(
            natural_generator,
            expected_generator,
            display_factory_class,
            dimension,
        )

        self.eta_generator = expected_generator  # Fix typing
        self.distribution_class = distribution_class

    def t(self, x: ArrayLike) -> Array:
        r""":math:`t(x)` sufficient statistics function.

        Args:
            x: Sample space input.

        Returns:
            Sufficient statistics function evaluated at x.
        """
        return self.distribution_class.t(x)

    def kl_divergence(self, point_1: Point, point_2: Point) -> Array:
        """KL-Divergence of two points in an exponential family manifold.

        This is equivalent to the Bregman divergence of natural or expected
        parameters.

        Args:
            point_1: Left-sided argument of the KL-Divergence.
            point_2: Right-sided argument of the KL-Divergence.

        Returns:
            KL-Divergence between point_1 and point_2 on the exponential family manifold.
        """
        breg_div = BregmanDivergence(self)
        return breg_div(point_1, point_2)
