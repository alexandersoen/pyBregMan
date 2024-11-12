import jax.numpy as jnp
from jax import Array

from bregman.application.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution,
    ExponentialFamilyManifold,
)
from bregman.base import THETA_COORDS, DisplayPoint, Point
from bregman.manifold.generator import AutoDiffGenerator


class GaussianPoint(DisplayPoint):
    """Display point for the Gaussian manifold."""

    @property
    def dimension(self) -> int:
        """Covariance matrix shape dimension.

        Returns:
            Covariance shape dimension (i.e, d in dxd covariance matrix).
        """
        return int(0.5 * (jnp.sqrt(4 * len(self.data) + 1) - 1))

    @property
    def mu(self) -> Array:
        """Mean value from point data.

        Returns:
            Mean of Gaussian distribution corresponding to the point.
        """
        return self.data[: self.dimension]

    @property
    def Sigma(self) -> Array:
        """Covariance value from point data.

        Returns:
            Covariance of Gaussian distribution corresponding to the point.
        """
        return self.data[self.dimension :].reshape(self.dimension, self.dimension)

    def display(self) -> str:
        """Generated pretty printed string on display.

        Returns:
            String of probability values of Gaussian point.
        """
        return f"$\\mu$ = {self.mu}; $\\Sigma$ = {self.Sigma}"


class GaussianDistribution(ExponentialFamilyDistribution):
    """Gaussian distributions as exponential family distributions.

    Attributes:
        dimension: Covariance shape dimension (i.e, d in dxd covariance matrix).
    """

    def __init__(self, theta: Array, dimension: int) -> None:
        """Initialize Gaussian distribution.

        Args:
            theta: Natural parameters of Gaussian distribution.
            dimension: Covariance shape dimension (i.e, d in dxd covariance matrix).
        """
        super().__init__(theta, (dimension,))

    @staticmethod
    def t(x: Array) -> Array:
        r""":math:`t(x)` sufficient statistics function of the Gaussian
        distribution.

        Args:
            x: Sample space input.

        Returns:
            Sufficient statistics function of the Gaussian distribution evaluated at x.
        """
        return jnp.concatenate([x, -jnp.outer(x, x).flatten()])

    @staticmethod
    def k(x: Array) -> Array:
        r""":math:`k(x)` carrier measure of the Gaussian distribution.

        Args:
            x: Sample space input.

        Returns:
            Carries measure of the Gaussian distribution evaluated at x.
        """
        return jnp.array(0.0)

    def F(self, x: Array) -> Array:
        r""":math:`F(x) = \log \int \exp(\theta^T t(x)) \mathrm{d}x`
        normalizer of the Gaussian distribution.

        Args:
            x: Parameter value.

        Returns:
            Normalizer of the Gaussian distribution evaluated at parameter value x.
        """
        theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension[0], x)

        return 0.5 * (
            0.5 * theta_mu.T @ jnp.linalg.inv(theta_sigma) @ theta_mu
            - jnp.log(jnp.linalg.det(theta_sigma))
            + self.dimension[0] * jnp.log(jnp.pi)
        )


class UnivariateGaussianDistribution(GaussianDistribution):
    """Univariate Gaussian distribution as an exponential family distribution."""

    def __init__(self, theta: Array) -> None:
        """Initialize univariate Gaussian distribution.

        Args:
            theta: Natural parameters of Gaussian distribution.
        """
        super().__init__(theta, 1)

    @staticmethod
    def t(x: Array) -> Array:
        r""":math:`t(x)` sufficient statistics function of the univariate
        Gaussian distribution.

        Args:
            x: Sample space input.

        Returns:
            Sufficient statistics function of the univariate Gaussian distribution evaluated at x.
        """
        return jnp.concatenate([x, jnp.outer(x, x).flatten()])

    @staticmethod
    def k(x: Array) -> Array:
        r""":math:`k(x)` carrier measure of the univariate Gaussian
        distribution.

        Args:
            x: Sample space input.

        Returns:
            Carries measure of the univariate Gaussian distribution evaluated at x.
        """
        return jnp.array(0.0)

    def F(self, x: Array) -> Array:
        r""":math:`F(x) = \log \int \exp(\theta^T t(x)) \mathrm{d}x` normalizer
        of the univariate Gaussian distribution.

        Args:
            x: Parameter value.

        Returns:
            Normalizer of the univariate Gaussian distribution evaluated at parameter value x.
        """

        theta_mu, theta_sigma = x

        return -0.25 * theta_mu * theta_mu / theta_sigma + 0.5 * jnp.log(
            -jnp.pi / theta_sigma
        )


class GaussianPrimalGenerator(AutoDiffGenerator):
    """Gaussian manifold primal Bregman generator."""

    def _F(self, x: Array) -> Array:
        """Gaussian manifold primal Bregman generator function.

        Args:
            x: Input value.

        Returns:
            Gaussian manifold primal Bregman generator value evaluated at x.
        """

        if self.dimension > 1:

            theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

            return 0.5 * (
                0.5 * theta_mu.T @ jnp.linalg.inv(theta_sigma) @ theta_mu
                - jnp.log(jnp.linalg.det(theta_sigma))
                + self.dimension * jnp.log(jnp.pi)
            )
        else:

            theta_mu, theta_sigma = x

            return (
                -0.25 * theta_mu * theta_mu / theta_sigma
                + 0.5 * jnp.log(jnp.pi)
                - 0.5 * jnp.log(-theta_sigma)
            )


class GaussianDualGenerator(AutoDiffGenerator):
    """Gaussian manifold dual Bregman generator."""

    def _F(self, x: Array) -> Array:
        """Gaussian manifold dual Bregman generator function.

        Args:
            x: Input value.

        Returns:
            Gaussian manifold dual Bregman generator value evaluated at x.
        """

        if self.dimension > 1:
            eta_mu, eta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

            return (
                -0.5 * jnp.log(1 + eta_mu.T @ jnp.linalg.inv(eta_sigma) @ eta_mu)
                - 0.5 * jnp.log(jnp.linalg.det(-eta_sigma))
                - 0.5 * self.dimension * (1 + jnp.log(2 * jnp.pi))
            )
        else:
            eta_mu, eta_sigma = x

            return -0.5 * jnp.log(jnp.abs(eta_mu * eta_mu - eta_sigma))


class GaussianManifold(ExponentialFamilyManifold[GaussianPoint, GaussianDistribution]):
    """Gaussian exponential family manifold.

    Attributes:
        input_dimension: Dimension of the sample space.
    """

    def __init__(self, input_dimension: int):
        """Initialize Gaussian manifold.

        Args:
            input_dimension: Dimension of the sample space.
        """
        F_gen = GaussianPrimalGenerator(input_dimension)
        G_gen = GaussianDualGenerator(input_dimension)

        self.input_dimension = input_dimension

        if input_dimension == 1:
            dist_class = UnivariateGaussianDistribution
        else:
            dist_class = GaussianDistribution

        super().__init__(
            natural_generator=F_gen,
            expected_generator=G_gen,
            distribution_class=dist_class,
            display_factory_class=GaussianPoint,
            dimension=input_dimension * (input_dimension + 1),
        )

    def point_to_distribution(self, point: Point) -> GaussianDistribution:
        """Converts a point to a Gaussian distribution.

        Args:
            point: Point to be converted.

        Returns:
            Gaussian distribution corresponding to the point.
        """
        theta = self.convert_coord(THETA_COORDS, point).data

        return GaussianDistribution(theta, self.input_dimension)

    def distribution_to_point(
        self, distribution: GaussianDistribution
    ) -> GaussianPoint:
        """Converts a Gaussian distribution to a point in the manifold.

        Args:
            distribution: Gaussian distribution to be converted.

        Returns:
            Point corresponding to the Gaussian distribution.
        """
        opoint = Point(
            coords=THETA_COORDS,
            data=distribution.theta,
        )
        return self.display_factory_class(opoint)

    def _lambda_to_theta(self, lamb: Array) -> Array:
        if self.input_dimension > 1:
            mu, Sigma = _flatten_to_mu_Sigma(self.input_dimension, lamb)
            inv_Sigma = jnp.linalg.inv(Sigma)

            theta_mu = inv_Sigma @ mu
            theta_Sigma = 0.5 * inv_Sigma

            return jnp.concatenate([theta_mu, theta_Sigma.flatten()])
        else:
            mu, sigma = lamb

            return jnp.array([mu / sigma, -0.5 / sigma])

    def _lambda_to_eta(self, lamb: Array) -> Array:
        if self.input_dimension > 1:
            mu, Sigma = _flatten_to_mu_Sigma(self.input_dimension, lamb)

            eta_mu = mu
            eta_Sigma = -Sigma - jnp.outer(mu, mu)

            return jnp.concatenate([eta_mu, eta_Sigma.flatten()])
        else:
            mu, sigma = lamb

            return jnp.array([mu, mu * mu + sigma])

    def _theta_to_lambda(self, theta: Array) -> Array:
        if self.input_dimension > 1:
            theta_mu, theta_Sigma = _flatten_to_mu_Sigma(self.input_dimension, theta)
            inv_theta_Sigma = jnp.linalg.inv(theta_Sigma)

            mu = 0.5 * inv_theta_Sigma @ theta_mu
            var = 0.5 * inv_theta_Sigma

            return jnp.concatenate([mu, var.flatten()])
        else:
            theta_mu, theta_sigma = theta

            return jnp.array([-0.5 * theta_mu / theta_sigma, -0.5 / theta_sigma])

    def _eta_to_lambda(self, eta: Array) -> Array:
        if self.input_dimension > 1:
            eta_mu, eta_Sigma = _flatten_to_mu_Sigma(self.input_dimension, eta)

            mu = eta_mu
            var = -eta_Sigma - jnp.outer(eta_mu, eta_mu)

            return jnp.concatenate([mu, var.flatten()])
        else:
            eta_mu, eta_sigma = eta

            return jnp.array([eta_mu, eta_sigma - eta_mu * eta_mu])


def _flatten_to_mu_Sigma(input_dimension: int, vec: Array) -> tuple[Array, Array]:
    mu = vec[:input_dimension]
    sigma = vec[input_dimension:].reshape(input_dimension, input_dimension)

    return mu, sigma
