from dataclasses import dataclass

import numpy as np

from bregman.base import Display, Point
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.application import ORDINARY_COORDS
from bregman.manifold.distribution.distribution import DistributionManifold
from bregman.object.distribution import Distribution


@dataclass
class Gaussian1DDisplay(Display):
    mu: float
    var: float

    def __repr__(self):
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

    def F(self, x: np.ndarray) -> np.ndarray:
        theta1 = x[0]
        theta2 = x[1]

        return 0.5 * (
            0.5 * np.power(theta1, 2) / theta2
            - np.log(np.abs(theta2))
            + np.log(np.pi)
        )


class Gaussian1DDualGenerator(AutoDiffGenerator):

    def F(self, x: np.ndarray) -> np.ndarray:
        eta1 = x[0]
        eta2 = x[1]

        return -0.5 * (
            np.log(1 + np.power(eta1, 2) / eta2)
            + np.log(np.abs(-eta2))
            + 1
            + np.log(2 * np.pi)
        )


class Gaussian1DManifold(
    DistributionManifold[Gaussian1DDisplay, Gaussian1DDistribution]
):

    def __init__(self):
        F_gen = Gaussian1DPrimalGenerator()
        G_gen = Gaussian1DDualGenerator()

        super().__init__(
            natural_generator=F_gen, expected_generator=G_gen, dimension=2
        )

    def point_to_distribution(self, point: Point) -> Gaussian1DDistribution:
        mu, var = self.convert_coord(ORDINARY_COORDS, point).data
        return Gaussian1DDistribution(mu, var)

    def distribution_to_point(
        self, distribution: Gaussian1DDistribution
    ) -> Point:
        return Point(
            coords=ORDINARY_COORDS,
            data=np.array([distribution.mu, distribution.var]),
        )

    def point_to_display(self, point: Point) -> Gaussian1DDisplay:
        mu, var = self.convert_coord(ORDINARY_COORDS, point).data
        return Gaussian1DDisplay(mu, var)

    def display_to_point(self, display: Gaussian1DDisplay) -> Point:
        return Point(
            coords=ORDINARY_COORDS, data=np.array([display.mu, display.var])
        )

    def _ordinary_to_natural(self, lamb: np.ndarray) -> np.ndarray:
        mu = lamb[0]
        var = lamb[1]

        theta1 = mu / var
        theta2 = 0.5 / var

        return np.array([theta1, theta2])

    def _ordinary_to_moment(self, lamb: np.ndarray) -> np.ndarray:
        mu = lamb[0]
        var = lamb[1]

        eta1 = mu
        eta2 = -var - np.power(mu, 2)

        return np.array([eta1, eta2])

    def _natural_to_ordinary(self, theta: np.ndarray) -> np.ndarray:
        theta1 = theta[0]
        theta2 = theta[1]

        mu = 0.5 * theta1 / theta2
        var = 0.5 / theta2

        return np.array([mu, var])

    def _moment_to_ordinary(self, eta: np.ndarray) -> np.ndarray:
        eta1 = eta[0]
        eta2 = eta[1]

        mu = eta1
        var = -eta2 - np.power(eta1, 2)

        return np.array([mu, var])
