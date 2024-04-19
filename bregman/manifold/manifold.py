from abc import ABC, abstractmethod

import numpy as np

from bregman.generator.generator import Bregman, Generator


class Geodesic:

    def __init__(
        self, F: Generator, source: np.ndarray, dest: np.ndarray
    ) -> None:
        self.F = F
        self.source = source
        self.dest = dest

    def path(self, t: float) -> np.ndarray:
        assert 0 <= t <= 1

        src_grad_F = self.F.grad(self.source)
        dst_grad_F = self.F.grad(self.dest)

        return self.F.grad_inv((1 - t) * src_grad_F + t * dst_grad_F)

    def __call__(self, t: float) -> np.ndarray:
        return self.path(t)

    def tangent(self, t: float) -> np.ndarray:
        grad_diff = self.F.grad(self.dest) - self.F.grad(self.source)
        rescale = np.linalg.inv(self.F.hess(self.path(t)))

        return rescale @ grad_diff


class Manifold(ABC):

    bregman: Bregman

    @abstractmethod
    def coord_to_natural(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def coord_to_moment(self, x: np.ndarray) -> np.ndarray:
        pass

    def natural_to_moment(self, theta: np.ndarray) -> np.ndarray:
        return self.bregman.F_generator.grad(theta)

    def moment_to_natural(self, eta: np.ndarray) -> np.ndarray:
        return self.bregman.G_generator.grad(eta)

    def primal_geodesic(
        self, theta_1: np.ndarray, theta_2: np.ndarray
    ) -> Geodesic:
        return Geodesic(self.bregman.F_generator, theta_1, theta_2)

    def dual_geodesic(self, eta_1: np.ndarray, eta_2: np.ndarray) -> Geodesic:
        return Geodesic(self.bregman.G_generator, eta_1, eta_2)


#    @abstractmethod
#    def tangent_vector_primal_geodesic(self):
#        pass
#
#    @abstractmethod
#    def tangent_vector_dual_geodesic(self):
#        pass
