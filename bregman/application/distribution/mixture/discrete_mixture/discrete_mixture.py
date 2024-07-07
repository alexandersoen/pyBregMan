from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bregman.application.distribution.mixture.mixture import MixtureManifold
from bregman.base import ETA_COORDS, THETA_COORDS, Point, Shape
from bregman.object.distribution import Distribution

if TYPE_CHECKING:
    from bregman.application.distribution.exponential_family.categorical import \
        CategoricalManifold


class DeltaDistribution(Distribution):

    def __init__(self, x: np.ndarray) -> None:
        super().__init__()

        self.dimension: Shape = (1,)
        self.x = x

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (x == self.x).astype(float)


class DiscreteMixtureManifold(MixtureManifold[DeltaDistribution]):

    def __init__(self, xs: list[np.ndarray]) -> None:
        distributions = [DeltaDistribution(x) for x in xs]

        super().__init__(distributions)

    def to_categorical_manifold(self) -> CategoricalManifold:
        from bregman.application.distribution.exponential_family.categorical import \
            CategoricalManifold

        return CategoricalManifold(
            len(self.distributions), [d.x for d in self.distributions]
        )

    def point_to_categorical_point(self, point: Point) -> Point:
        eta_point = self.convert_coord(ETA_COORDS, point)
        return Point(THETA_COORDS, eta_point.data)
