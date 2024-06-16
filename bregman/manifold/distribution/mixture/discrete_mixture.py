from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bregman.base import Shape
from bregman.manifold.distribution.exponential_family.multinomial import (
    MultinomialDualGenerator, MultinomialPrimalGenerator)
from bregman.manifold.distribution.mixture.mixture import MixtureManifold
from bregman.object.distribution import Distribution

if TYPE_CHECKING:
    from bregman.manifold.distribution.exponential_family.categorical import \
        CategoricalManifold


class DeltaDistribution(Distribution):

    def __init__(self, x: np.ndarray) -> None:
        super().__init__()

        self.dimension: Shape = (1,)
        self.x = x

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (x == self.x).astype(float)


class DiscreteMixture(MixtureManifold[DeltaDistribution]):

    def __init__(self, xs: list[np.ndarray]) -> None:
        distributions = [DeltaDistribution(x) for x in xs]

        super().__init__(distributions)

    def to_categorical_manifold(self) -> CategoricalManifold:
        from bregman.manifold.distribution.exponential_family.categorical import \
            CategoricalManifold

        return CategoricalManifold(
            len(self.distributions), [d.x for d in self.distributions]
        )
