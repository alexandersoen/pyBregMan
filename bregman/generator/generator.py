from abc import ABC, abstractmethod

import numpy as np


class Generator(ABC):

    @abstractmethod
    def F(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def F_grad(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def F_hess(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def G(self, eta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def G_grad(self, eta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def G_hess(self, eta: np.ndarray) -> np.ndarray:
        pass

    def primal_divergence(self, theta_1, theta_2):
        return (
            self.F(theta_1)
            - self.F(theta_2)
            - np.inner(self.F_grad(theta_2), theta_1 - theta_2)
        )

    def dual_divergence(self, eta_1, eta_2):
        return (
            self.G(eta_1)
            - self.G(eta_2)
            - np.inner(self.F_grad(eta_2), eta_1 - eta_2)
        )
