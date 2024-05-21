from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np

from bregman.base import DisplayPoint, Point, Shape
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import LAMBDA_COORDS, point_convert_wrapper
from bregman.manifold.distribution.distribution import DistributionManifold
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


class GaussianDistribution(Distribution):

    def __init__(
        self, mu: np.ndarray, sigma: np.ndarray, dimension: int
    ) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma

        self.dimension: Shape = (dimension,)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        const = np.power(2 * np.pi, self.dimension[0] / 2) * np.sqrt(
            np.linalg.det(self.sigma)
        )
        mean_diff = x - self.mu

        return (
            np.exp(
                -0.5 * (mean_diff.T @ np.linalg.inv(self.sigma) @ mean_diff)
            )
            / const
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
    DistributionManifold[GaussianPoint, GaussianDistribution]
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
        ordinary_point = self.convert_to_display(point)
        return GaussianDistribution(
            ordinary_point.mu, ordinary_point.Sigma, self.input_dimension
        )

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
