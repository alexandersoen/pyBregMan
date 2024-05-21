from dataclasses import dataclass

import numpy as np

from bregman.base import DisplayPoint, Point
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import LAMBDA_COORDS, point_convert_wrapper
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.object.distribution import Distribution


class Gaussian1DPoint(DisplayPoint):
    @property
    def mu(self):
        return self.data[0]

    @property
    def var(self):
        return self.data[1]

    def display(self):
        return f"$\\mu = {self.mu}; \\sigma^2 = {self.var}$"


class Gaussian1DDistribution(Distribution):

    def __init__(self, mu: float, var: float) -> None:
        super().__init__()

        self.mu = mu
        self.var = var

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(-0.5 * np.power(x - self.mu, 2) / self.var)) / (
            np.sqrt(2 * self.var * np.pi)
        )


class Gaussian1DPrimalGenerator(AutoDiffGenerator):

    def _F(self, x: np.ndarray) -> np.ndarray:
        theta1 = x[0]
        theta2 = x[1]

        return 0.5 * (
            0.5 * np.power(theta1, 2) / theta2
            - np.log(np.abs(theta2))
            + np.log(np.pi)
        )


class Gaussian1DDualGenerator(AutoDiffGenerator):

    def _F(self, x: np.ndarray) -> np.ndarray:
        eta1 = x[0]
        eta2 = x[1]

        return -0.5 * (
            np.log(1 + np.power(eta1, 2) / eta2)
            + np.log(np.abs(-eta2))
            + 1
            + np.log(2 * np.pi)
        )


class Gaussian1DManifold(
    DistributionManifold[Gaussian1DPoint, Gaussian1DDistribution]
):

    def __init__(self):
        F_gen = Gaussian1DPrimalGenerator()
        G_gen = Gaussian1DDualGenerator()

        super().__init__(
            natural_generator=F_gen,
            expected_generator=G_gen,
            display_factory=point_convert_wrapper(Gaussian1DPoint),
            dimension=2,
        )

    def point_to_distribution(self, point: Point) -> Gaussian1DDistribution:
        mu, var = self.convert_coord(LAMBDA_COORDS, point).data
        return Gaussian1DDistribution(mu, var)

    def distribution_to_point(
        self, distribution: Gaussian1DDistribution
    ) -> Gaussian1DPoint:
        return self.convert_to_display(
            Point(
                coords=LAMBDA_COORDS,
                data=np.array([distribution.mu, distribution.var]),
            )
        )

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        mu = lamb[0]
        var = lamb[1]

        theta1 = mu / var
        theta2 = 0.5 / var

        return np.array([theta1, theta2])

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        mu = lamb[0]
        var = lamb[1]

        eta1 = mu
        eta2 = -var - np.power(mu, 2)

        return np.array([eta1, eta2])

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        theta1 = theta[0]
        theta2 = theta[1]

        mu = 0.5 * theta1 / theta2
        var = 0.5 / theta2

        return np.array([mu, var])

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        eta1 = eta[0]
        eta2 = eta[1]

        mu = eta1
        var = -eta2 - np.power(eta1, 2)

        return np.array([mu, var])
