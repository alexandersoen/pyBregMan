import numpy as np

from bregman.generator.generator import Generator
from bregman.manifold.application import ApplicationManifold


class Gaussian1DPrimalGenerator(Generator):

    def __init__(self):
        pass

    def F(self, x: np.ndarray) -> np.ndarray:
        theta1 = x[0]
        theta2 = x[1]

        return 0.5 * (
            0.5 * np.power(theta1, 2) / theta2
            - np.log(np.abs(theta2))
            + np.log(np.pi)
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        theta1 = x[0]
        theta2 = x[1]

        eta1 = 0.5 * theta1 / theta2
        eta2 = -0.5 * (1 / theta2 + 0.5 * np.power(theta1 / theta2, 2))

        return np.array([eta1, eta2])

    def hess(self, x: np.ndarray) -> np.ndarray:
        theta1 = x[0]
        theta2 = x[1]

        g11 = 0.5 / theta2
        g22 = 0.5 * (np.power(theta1, 2) + theta2) / np.power(theta2, 3)
        g12 = -0.5 * theta1 / np.power(theta2, 2)
        g21 = g12

        return np.array([[g11, g12], [g21, g22]])

    def grad_inv(self, x: np.ndarray) -> np.ndarray:
        eta1 = x[0]
        eta2 = x[1]

        theta1 = -eta1 / (eta2 + np.power(eta1, 2))
        theta2 = -0.5 / (eta2 + np.power(eta1, 2))

        return np.array([theta1, theta2])


class Gaussian1DDualGenerator(Generator):

    def __init__(self):
        pass

    def F(self, x: np.ndarray) -> np.ndarray:
        eta1 = x[0]
        eta2 = x[1]

        return -0.5 * (
            np.log(1 + np.power(eta1, 2) / eta2)
            + np.log(np.abs(-eta2))
            + 1
            + np.log(2 * np.pi)
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        eta1 = x[0]
        eta2 = x[1]

        theta1 = -eta1 / (eta2 + np.power(eta1, 2))
        theta2 = -0.5 / (eta2 + np.power(eta1, 2))

        return np.array([theta1, theta2])

    def hess(self, x: np.ndarray) -> np.ndarray:
        eta1 = x[0]
        eta2 = x[1]

        g11 = (np.power(eta1, 2) - eta2) / np.power(
            (np.power(eta1, 2) + eta2), 2
        )
        g22 = 0.5 / np.power((np.power(eta1, 2) + eta2), 2)
        g12 = eta1 / np.power((np.power(eta1, 2) + eta2), 2)
        g21 = g12

        return np.array([[g11, g12], [g21, g22]])

    def grad_inv(self, x: np.ndarray) -> np.ndarray:
        theta1 = x[0]
        theta2 = x[1]

        eta1 = 0.5 * theta1 / theta2
        eta2 = -0.5 * (1 / theta2 + 0.5 * np.power(theta1 / theta2, 2))

        return np.array([eta1, eta2])


class Gaussian1DManifold(ApplicationManifold):

    def __init__(self):
        F_gen = Gaussian1DPrimalGenerator()
        G_gen = Gaussian1DDualGenerator()

        super().__init__(
            natural_generator=F_gen, expected_generator=G_gen, dimension=2
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
