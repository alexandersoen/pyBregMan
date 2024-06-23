import numpy as np

from bregman.base import LAMBDA_COORDS, Point
from bregman.dissimilarity.base import Dissimilarity

from .multinomial import MultinomialManifold


class FisherRaoMultinomialDistance(Dissimilarity[MultinomialManifold]):

    def __init__(self, manifold: MultinomialManifold) -> None:
        super().__init__(manifold)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:

        probs_1 = self.manifold.convert_coord(LAMBDA_COORDS, point_1).data
        probs_2 = self.manifold.convert_coord(LAMBDA_COORDS, point_2).data

        return (
            2
            * np.sqrt(self.manifold.k)
            * np.arccos(np.sum(np.sqrt(probs_1 * probs_2)))
        )
