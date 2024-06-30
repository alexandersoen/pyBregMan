import numpy as np

from bregman.application.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.base import LAMBDA_COORDS, DualCoords, Point
from bregman.dissimilarity.bregman import BregmanDivergence


# Define manifold + objects
def test_bregman_divergence_gaussian():
    manifold = GaussianManifold(2)

    coord1 = Point(LAMBDA_COORDS, np.array([0, 0, 1, 0.5, 0.5, 2]))
    coord2 = Point(LAMBDA_COORDS, np.array([1, 2, 1, 0, 0, 0.5]))

    primal_div = BregmanDivergence(manifold, dcoords=DualCoords.THETA)
    dual_div = BregmanDivergence(manifold, dcoords=DualCoords.ETA)

    print(primal_div(coord1, coord2), dual_div(coord2, coord1))
    assert np.isclose(primal_div(coord1, coord2), dual_div(coord2, coord1))
