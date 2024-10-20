from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array
from jax.typing import ArrayLike

from bregman.application.distribution.mixture.mixture import MixtureManifold
from bregman.base import ETA_COORDS, THETA_COORDS, Point
from bregman.object.distribution import Distribution

if TYPE_CHECKING:
    from bregman.application.distribution.exponential_family.categorical import (
        CategoricalManifold,
    )


class DeltaDistribution(Distribution):
    """Dirac / Indicator "distribution".

    Parameters:
        x: Point at Indicator.
    """

    def __init__(self, x: ArrayLike) -> None:
        """Initialize Indicator distribution.

        Args:
            x: Point at Indicator.
        """
        super().__init__((1,))
        self.x = x

    def pdf(self, x: ArrayLike) -> Array:
        """Evaluate p.d.f. of Indicator distribution.

        Args:
            x: Point in sample space.

        Returns:
            P.d.f. of Indicator distribution.
        """
        return (x == self.x).astype(float)


class DiscreteMixtureManifold(MixtureManifold[DeltaDistribution]):
    """Discrete Mixture manifold. Mixing components are set to be indicator /
    Dirac distributions.
    """

    def __init__(self, xs: list[ArrayLike]) -> None:
        """Initialize Discrete Mixture manifold.

        Args:
            xs: Values to create Dirac mixing components.
        """
        distributions = [DeltaDistribution(x) for x in xs]

        super().__init__(distributions)

    def to_categorical_manifold(self) -> CategoricalManifold:
        """The corresponding dual Categorical manifold.

        Returns:
            Categorical manifold dual to the Mixture manifold.
        """
        from bregman.application.distribution.exponential_family.categorical import (
            CategoricalManifold,
        )

        return CategoricalManifold(
            len(self.distributions), [d.x for d in self.distributions]
        )

    def point_to_categorical_point(self, point: Point) -> Point:
        """Convert a point in the Mixture manifold into the dual point in
        the Categorical manifold.

        Args:
            point: Point parameterized in the Mixture manifold.

        Returns:
            Point parameterized in the Categorical manifold.
        """
        eta_point = self.convert_coord(ETA_COORDS, point)
        return Point(THETA_COORDS, eta_point.data)
