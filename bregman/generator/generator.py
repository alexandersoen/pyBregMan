from abc import ABC, abstractmethod

import autograd
import numpy as np


class Generator(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def F(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hess(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.F(x)


class AutoDiffGenerator(Generator, ABC):

    def __init__(self):
        super().__init__()

    def grad(self, x: np.ndarray) -> np.ndarray:
        return autograd.grad(self.F, x)

    def hess(self, x: np.ndarray) -> np.ndarray:
        return autograd.hessian(self.F, x)


class Bregman:

    def __init__(self, F_generator: Generator, G_generator: Generator) -> None:
        self.F_generator = F_generator
        self.G_generator = G_generator

    def F_divergence(self, theta_1, theta_2):
        return (
            self.F_generator(theta_1)
            - self.F_generator(theta_2)
            - np.inner(self.F_generator.grad(theta_2), theta_1 - theta_2)
        )

    def G_divergence(self, eta_1, eta_2):
        return (
            self.G_generator(eta_1)
            - self.G_generator(eta_2)
            - np.inner(self.G_generator.grad(eta_2), eta_1 - eta_2)
        )
