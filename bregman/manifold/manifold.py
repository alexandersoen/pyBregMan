from abc import ABC, abstractmethod

from bregman.generator.generator import Generator


class manifold(ABC):

    generator: Generator

    @abstractmethod
    def primal_geodesic(self):
        pass

    @abstractmethod
    def dual_geodesic(self):
        pass

    @abstractmethod
    def tangent_vector_primal_geodesic(self):
        pass

    @abstractmethod
    def tangent_vector_dual_geodesic(self):
        pass
