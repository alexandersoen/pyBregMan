from typing import Generic, Sequence, TypeVar

from bregman.application.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution, ExponentialFamilyManifold)
from bregman.application.distribution.mixture.mixture import (
    MixtureDistribution, MixtureManifold)
from bregman.base import LAMBDA_COORDS, DisplayPoint, Point

EFDistribution = TypeVar("EFDistribution", bound=ExponentialFamilyDistribution)
EFDisplay = TypeVar("EFDisplay", bound=DisplayPoint)
EFManifold = TypeVar("EFManifold", bound=ExponentialFamilyManifold)


class EFMixtureManifold(
    MixtureManifold,
    Generic[EFDisplay, EFDistribution],
):

    def __init__(
        self,
        distributions: Sequence[EFDistribution],
        ef_manifold: ExponentialFamilyManifold[EFDisplay, EFDistribution],
    ) -> None:
        super().__init__(distributions)

        self.ef_manifold = ef_manifold

    def weight_distributions_to_distribution(
        self, w_point: Point, d_points: list[Point]
    ) -> MixtureDistribution:
        assert len(w_point.data) == self.dimension + 1
        assert len(d_points) == self.dimension + 1

        weights = self.convert_coord(LAMBDA_COORDS, w_point).data
        distributions = [
            self.ef_manifold.point_to_distribution(
                self.ef_manifold.convert_to_display(p)
            )
            for p in d_points
        ]

        return MixtureDistribution(weights, distributions)
