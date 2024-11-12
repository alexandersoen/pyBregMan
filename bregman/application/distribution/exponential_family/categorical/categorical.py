from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from bregman.application.distribution.exponential_family.multinomial import (
    MultinomialManifold,
)
from bregman.base import ETA_COORDS, THETA_COORDS, Point

if TYPE_CHECKING:
    from bregman.application.distribution.mixture.discrete_mixture import (
        DiscreteMixtureManifold,
    )


class NumberOfCategoriesMissMatch(Exception):
    """Exception when categories to be set are miss-matched from number of
    categories defined for the manifold.
    """

    pass


class CategoricalManifold(MultinomialManifold):
    """Categorical exponential family manifold.

    Parameters:
        categories: Categorical choices for the distributions.
    """

    def __init__(self, k: int, categories: list[Array] | None = None) -> None:
        """Initialize Categorical manifold.

        Args:
            k: Number of categories.
            categories: Vectors corresponding to categories.
        """
        super().__init__(k, n=1)

        if categories is None:
            categories = [v for v in jnp.eye(k)]

        self.set_categories(categories)

    def set_categories(self, categories: list[Array]) -> None:
        """Set the categorical indices to specific category vectors.

        Args:
            categories: Categories to be set.

        Raises:
            NumberOfCategoriesMissMatch: When number of categories do no match.
        """
        if len(categories) != self.k:
            raise NumberOfCategoriesMissMatch()

        self.categories = categories

    def to_discrete_mixture_manifold(self) -> DiscreteMixtureManifold:
        """The corresponding dual Mixture manifold.

        Returns:
            Mixture manifold dual to the Categorical manifold.
        """
        from bregman.application.distribution.mixture.discrete_mixture import (
            DiscreteMixtureManifold,
        )

        return DiscreteMixtureManifold(self.categories)

    def point_to_mixture_point(self, point: Point) -> Point:
        """Convert a point in the Categorical manifold into the dual point in
        the Mixture manifold.

        Args:
            point: Point parameterized in the Categorical manifold.

        Returns:
            Point parameterized in the Mixture manifold.
        """
        eta_point = self.convert_coord(ETA_COORDS, point)
        return Point(THETA_COORDS, eta_point.data)
