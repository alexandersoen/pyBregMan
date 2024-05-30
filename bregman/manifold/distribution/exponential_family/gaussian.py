from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np

from bregman.base import DisplayPoint, Point, Shape
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import LAMBDA_COORDS, point_convert_wrapper
from bregman.manifold.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution, ExponentialFamilyManifold)
from bregman.manifold.manifold import THETA_COORDS
from bregman.object.distribution import Distribution


class GaussianPoint(DisplayPoint):

    @property
    def dimension(self) -> int:
        return int(0.5 * (np.sqrt(4 * len(self.data) + 1) - 1))

    @property
    def mu(self) -> np.ndarray:
        return self.data[: self.dimension]

    @property
    def Sigma(self) -> np.ndarray:
        return self.data[self.dimension :].reshape(
            self.dimension, self.dimension
        )

    def display(self) -> str:
        return f"$\\mu$ = {self.mu}; $\\Sigma$ = {self.Sigma}"


class GaussianDistribution(ExponentialFamilyDistribution):

    def __init__(self, theta: np.ndarray, dimension: int) -> None:
        super().__init__()

        self.theta = theta

        self.dimension: Shape = (dimension,)

    def t(self, x: np.ndarray) -> np.ndarray:
        r"""t(x) sufficient statistics function."""
        return np.concatenate([x, -np.outer(x, x).flatten()])

    def k(self, x: np.ndarray) -> np.ndarray:
        r"""k(x) carrier measure."""
        return np.array(0.0)

    def F(self, x: np.ndarray) -> np.ndarray:
        r"""F(x) = \log \int \exp(\theta^\T t(x)) dx normalizer"""
        theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension[0], x)

        return 0.5 * (
            0.5 * theta_mu.T @ np.linalg.inv(theta_sigma) @ theta_mu
            - np.log(np.linalg.det(theta_sigma))
            + self.dimension * np.log(np.pi)
        )


class GaussianPrimalGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

    def _F(self, x: np.ndarray) -> np.ndarray:
        theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

        return 0.5 * (
            0.5 * theta_mu.T @ anp.linalg.inv(theta_sigma) @ theta_mu
            - anp.log(anp.linalg.det(theta_sigma))
            + self.dimension * anp.log(np.pi)
        )


class GaussianDualGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

    def _F(self, x: np.ndarray) -> np.ndarray:
        eta_mu, eta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

        return -0.5 * (
            anp.log(1 + eta_mu.T @ anp.linalg.inv(eta_sigma) @ eta_mu)
            + anp.log(anp.linalg.det(-eta_sigma))
            + self.dimension * (1 + anp.log(2 * np.pi))
        )


class GaussianManifold(
    ExponentialFamilyManifold[GaussianPoint, GaussianDistribution]
):

    def __init__(self, input_dimension: int):
        F_gen = GaussianPrimalGenerator(input_dimension)
        G_gen = GaussianDualGenerator(input_dimension)

        self.input_dimension = input_dimension

        super().__init__(
            natural_generator=F_gen,
            expected_generator=G_gen,
            display_factory=point_convert_wrapper(GaussianPoint),
            dimension=input_dimension * (input_dimension + 1),
        )

    def point_to_distribution(self, point: Point) -> GaussianDistribution:
        theta = self.convert_coord(THETA_COORDS, point)

        return GaussianDistribution(theta, self.input_dimension)

    def distribution_to_point(
        self, distribution: GaussianDistribution
    ) -> GaussianPoint:
        opoint = Point(
            coords=LAMBDA_COORDS,
            data=np.concatenate(
                [distribution.mu, distribution.sigma.flatten()]
            ),
        )
        return self.display_factory(opoint)

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        mu, Sigma = _flatten_to_mu_Sigma(self.input_dimension, lamb)
        inv_Sigma = np.linalg.inv(Sigma)

        theta_mu = inv_Sigma @ mu
        theta_Sigma = 0.5 * inv_Sigma

        return np.concatenate([theta_mu, theta_Sigma.flatten()])

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        mu, Sigma = _flatten_to_mu_Sigma(self.input_dimension, lamb)

        eta_mu = mu
        eta_Sigma = -Sigma - mu @ mu.T

        return np.concatenate([eta_mu, eta_Sigma.flatten()])

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        theta_mu, theta_Sigma = _flatten_to_mu_Sigma(
            self.input_dimension, theta
        )
        inv_theta_Sigma = np.linalg.inv(theta_Sigma)

        mu = 0.5 * inv_theta_Sigma @ theta_mu
        var = 0.5 * inv_theta_Sigma

        return np.concatenate([mu, var.flatten()])

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        eta_mu, eta_Sigma = _flatten_to_mu_Sigma(self.input_dimension, eta)

        mu = eta_mu
        var = -eta_Sigma - eta_mu @ eta_mu.T

        return np.concatenate([mu, var.flatten()])


def _flatten_to_mu_Sigma(
    input_dimension: int, vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = vec[:input_dimension]
    sigma = vec[input_dimension:].reshape(input_dimension, input_dimension)

    return mu, sigma
