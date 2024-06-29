import numpy as np

from bregman.application.application import LAMBDA_COORDS
from bregman.application.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.base import Point
from bregman.dissimilarity.bregman import BregmanDivergence
from bregman.manifold.manifold import DualCoord

# Define manifold + objects
def test_bregman_divergence_gaussian():
    manifold = GaussianManifold(2)

    coord1 = Point(LAMBDA_COORDS, np.array([0, 0, 1, 0.5, 0.5, 2]))
    coord2 = Point(LAMBDA_COORDS, np.array([1, 2, 1, 0, 0, 0.5]))

    primal_div = BregmanDivergence(manifold, coord=DualCoord.THETA)
    dual_div = BregmanDivergence(manifold, coord=DualCoord.ETA)

    print(primal_div(coord1, coord2), dual_div(coord2, coord1))
    assert np.isclose(primal_div(coord1, coord2), dual_div(coord2, coord1))