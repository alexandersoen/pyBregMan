from abc import ABC, abstractmethod

from jax import Array

from bregman.base import Shape


class Distribution(ABC):
    """Abstract class for distributions.

    Parameters:
        dimension: Sample space dimension.
    """

    def __init__(self, dimension: Shape) -> None:
        super().__init__()

        self.dimension = dimension

    @abstractmethod
    def pdf(self, x: Array) -> Array:
        """Probability density function (p.d.f.) of distribution.

        Args:
            x: Input for p.d.f.

        Returns:
            P.d.f. value of the distribution evaluated at x.
        """
        pass
