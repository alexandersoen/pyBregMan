from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bregman.application.distribution.exponential_family.multinomial import \
    MultinomialManifold

if TYPE_CHECKING:
    from bregman.application.distribution.mixture.discrete_mixture import \
        DiscreteMixture


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

    def to_discrete_mixture_manifold(self) -> DiscreteMixture:

        from bregman.manifold.distribution.mixture.discrete_mixture import \
            DiscreteMixture

        return DiscreteMixture(self.categories)
