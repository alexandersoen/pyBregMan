from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bregman.application.distribution.exponential_family.multinomial import \
    MultinomialManifold
from bregman.base import ETA_COORDS, THETA_COORDS, Point

if TYPE_CHECKING:
    from bregman.application.distribution.mixture.discrete_mixture import \
        DiscreteMixtureManifold


class NumberOfCategoriesMissMatch(Exception):
    pass


class CategoricalManifold(MultinomialManifold):

    def __init__(
        self, k: int, categories: list[np.ndarray] | None = None
    ) -> None:
        super().__init__(k, n=1)

        if categories is None:
            categories = [v for v in np.eye(k)]
        self.categories = categories

    def set_categories(self, categories: list[np.ndarray]) -> None:
        if len(categories) != self.k:
            raise NumberOfCategoriesMissMatch()

        self.categories = categories

    def to_discrete_mixture_manifold(self) -> DiscreteMixtureManifold:

        from bregman.application.distribution.mixture.discrete_mixture import \
            DiscreteMixtureManifold

        return DiscreteMixtureManifold(self.categories)

    def point_to_mixture_point(self, point: Point) -> Point:
        eta_point = self.convert_coord(ETA_COORDS, point)
        return Point(THETA_COORDS, eta_point.data)
