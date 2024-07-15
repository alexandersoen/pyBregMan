import numpy as np

from bregman.base import LAMBDA_COORDS, Point
from bregman.dissimilarity.base import Dissimilarity

from .multinomial import MultinomialManifold


class FisherRaoMultinomialDistance(Dissimilarity[MultinomialManifold]):
    """Fisher-Rao distance on the Multinomial manifold."""

    def dissimilarity(self, point_1: Point, point_2: Point) -> np.ndarray:
        """Calculate Fisher-Rao distance for points on the Multinomial manifold.

        Args:
            point_1: Left-sided argument of the Fisher-Rao distance.
            point_2: Right-sided argument of the Fisher-Rao distance.

        Returns:
            Fisher-Rao distance between point_1 and point_2 on the Multinomial manifold.
        """
        probs_1 = self.manifold.convert_coord(LAMBDA_COORDS, point_1).data
        probs_2 = self.manifold.convert_coord(LAMBDA_COORDS, point_2).data

        return (
            2
            * np.sqrt(self.manifold.k)
            * np.arccos(np.sum(np.sqrt(probs_1 * probs_2)))
        )
