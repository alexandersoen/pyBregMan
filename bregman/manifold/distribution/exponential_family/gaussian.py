from dataclasses import dataclass

import numpy as np

from bregman.base import Display, Point, Shape
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import ORDINARY_COORDS
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.object.distribution import Distribution


@dataclass
class GaussianDisplay(Display):
    mu: np.ndarray
    Sigma: np.ndarray


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

    def F(self, x: np.ndarray) -> np.ndarray:
        theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

        return 0.5 * (
            0.5 * theta_mu.T @ np.linalg.inv(theta_sigma) @ theta_mu
            - np.log(np.linalg.det(theta_sigma))
            + self.dimension * np.log(np.pi)
        )


class GaussianDualGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

    def F(self, x: np.ndarray) -> np.ndarray:
        eta_mu, eta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

        return -0.5 * (
            np.log(1 + eta_mu.T @ np.linalg.inv(eta_sigma) @ eta_mu)
            + np.log(np.linalg.det(-eta_sigma))
            + self.dimension * (1 + np.log(2 * np.pi))
        )


class GaussianManifold(
    DistributionManifold[GaussianDisplay, GaussianDistribution]
):

    def __init__(self, dimension: int):
        F_gen = GaussianPrimalGenerator(dimension)
        G_gen = GaussianDualGenerator(dimension)

        super().__init__(
            natural_generator=F_gen, expected_generator=G_gen, dimension=2
        )

    def point_to_distribution(self, point: Point) -> GaussianDistribution:
        display = self.point_to_display(point)
        return GaussianDistribution(display.mu, display.Sigma, self.dimension)

    def distribution_to_point(
        self, distribution: GaussianDistribution
    ) -> Point:
        return Point(
            coords=ORDINARY_COORDS,
            data=np.concatenate(
                [distribution.mu, distribution.sigma.flatten()]
            ),
        )

    def point_to_display(self, point: Point) -> GaussianDisplay:
        odata = self.convert_coord(ORDINARY_COORDS, point).data
        mu, Sigma = _flatten_to_mu_Sigma(self.dimension, odata)

        return GaussianDisplay(mu=mu, Sigma=Sigma)

    def display_to_point(self, display: GaussianDisplay) -> Point:
        return Point(
            coords=ORDINARY_COORDS,
            data=np.concatenate([display.mu, display.Sigma.flatten()]),
        )

    def _ordinary_to_natural(self, lamb: np.ndarray) -> np.ndarray:
        mu, Sigma = _flatten_to_mu_Sigma(self.dimension, lamb)
        inv_Sigma = np.linalg.inv(Sigma)

        theta_mu = inv_Sigma @ mu
        theta_Sigma = 0.5 * inv_Sigma

        return np.concatenate([theta_mu, theta_Sigma.flatten()])

    def _ordinary_to_moment(self, lamb: np.ndarray) -> np.ndarray:
        mu, Sigma = _flatten_to_mu_Sigma(self.dimension, lamb)

        eta_mu = mu
        eta_Sigma = -Sigma - mu @ mu.T

        return np.concatenate([eta_mu, eta_Sigma.flatten()])

    def _natural_to_ordinary(self, theta: np.ndarray) -> np.ndarray:
        theta_mu, theta_Sigma = _flatten_to_mu_Sigma(self.dimension, theta)
        inv_theta_Sigma = np.linalg.inv(theta_Sigma)

        mu = 0.5 * inv_theta_Sigma @ theta_mu
        var = 0.5 * inv_theta_Sigma

        return np.concatenate([mu, var.flatten()])

    def _moment_to_ordinary(self, eta: np.ndarray) -> np.ndarray:
        eta_mu, eta_Sigma = _flatten_to_mu_Sigma(self.dimension, eta)

        mu = eta_mu
        var = -eta_Sigma - eta_mu @ eta_mu.T

        return np.concatenate([mu, var.flatten()])


def _flatten_to_mu_Sigma(
    dimension: int, vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = vec[:dimension]
    sigma = vec[dimension:].reshape(dimension, dimension)

    return mu, sigma
