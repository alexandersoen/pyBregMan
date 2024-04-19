import numpy as np

from bregman.generator.generator import Bregman, Generator
from bregman.manifold.manifold import Manifold


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


class Gaussian1DManifold(Manifold):

    def __init__(self):
        F_gen = Gaussian1DPrimalGenerator()
        G_gen = Gaussian1DDualGenerator()

        self.bregman = Bregman(F_gen, G_gen)

    def coord_to_natural(self, x: np.ndarray) -> np.ndarray:
        mu = x[0]
        var = x[1]

        theta1 = mu / var
        theta2 = 0.5 / var

        return np.array([theta1, theta2])

    def coord_to_moment(self, x: np.ndarray) -> np.ndarray:
        mu = x[0]
        var = x[1]

        eta1 = mu
        eta2 = -var - np.power(mu, 2)

        return np.array([eta1, eta2])
