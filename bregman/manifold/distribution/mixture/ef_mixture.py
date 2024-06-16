from typing import Generic, Sequence, TypeVar

import numpy as np

from bregman.base import LAMBDA_COORDS, DisplayPoint, Point
from bregman.manifold.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution, ExponentialFamilyManifold)
from bregman.manifold.distribution.mixture.mixture import (MixtureDistribution,
                                                           MixtureManifold)
from bregman.manifold.manifold import DualCoord

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

    """
    def soft_cluster(
        self,
        points_to_cluster: list[np.ndarray],
        init_weights: Point | None = None,
        init_distributions: Sequence[Point] | None = None,
        max_iter: int = 20,
    ):
        coords = DualCoord.THETA
        primal_gen = self.ef_manifold.bregman_generator(coords)
        dual_gen = self.ef_manifold.bregman_generator(coords.dual())

        cur_weights = init_weights
        cur_distributions = list(
            init_distributions if init_distributions else self.distributions
        )
        if cur_weights is None:
            cur_weights = Point(
                LAMBDA_COORDS,
                np.ones(self.dimension + 1) / (self.dimension + 1),
            )
        suff_func = self.ef_manifold.point_to_distribution(
            self.ef_manifold.convert_to_display(cur_distributions[0])
        ).t

        ll_new = sum(
            self.weight_distributions_to_distribution(
                cur_weights, cur_distributions
            ).pdf(x)
            for x in points_to_cluster
        )
        ll_threshold = ll_new * 0.001
        ll_old = float("inf")
        t = 0
        

        // Step E: computation of P (matrix of weights)
        double[][] p = new double[m][n];
        for (row=0; row<m; row++){
            double sum = 0;
            for (col=0; col<n; col++){
                double tmp  = fH.weight[col] * Math.exp( fL.EF.G(fH.param[col]) +  fL.EF.t(pointSet[row]).Minus(fH.param[col]).InnerProduct(fL.EF.gradG(fH.param[col])) );
                p[row][col] = tmp;
                sum        += tmp;
            }
            for (col=0; col<n; col++)
                p[row][col] /= sum;
        }

        // Step M: computation of parameters
        for (col=0; col<n; col++){
            double sum    = p[0][col];
            fH.param[col] = fL.EF.t(pointSet[0]).Times(p[0][col]);
            for (row=1; row<m; row++){
                sum          += p[row][col];
                fH.param[col] = fH.param[col].Plus( fL.EF.t(pointSet[row]).Times(p[row][col]) );
            }
            fH.weight[col]     = sum / m;
            fH.param[col]      = fH.param[col].Times(1./sum);
            fH.param[col].type = TYPE.EXPECTATION_PARAMETER;
        }

        

        while np.abs(ll_new - ll_old) > ll_threshold and t < max_iter:
            ll_old = ll_new

            dual_dist_params = np.stack(
                [
                    self.ef_manifold.convert_coord(
                        coords.dual().value, d_point
                    ).data
                    for d_point in cur_distributions
                ]
            )
            cur_weights * np.exp(
                np.apply_along_axis(dual_gen, 1, dual_dist_params)
                + np.einsum(
                    "ij,ij -> i",
                    suff_func(np.stack(points_to_cluster)) - dual_dist_params,
                    np.apply_along_axis(dual_gen.grad, 1, dual_dist_params),
                )
            )

            ll_new = sum(
                self.weight_distributions_to_distribution(
                    cur_weights, cur_distributions
                ).pdf(x)
                for x in points_to_cluster
            )

            t += 1
        """
