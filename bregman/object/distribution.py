from abc import ABC, abstractmethod

import numpy as np

from bregman.base import Shape


class Distribution(ABC):

    dimension: Shape

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass


#    @abstractmethod
#    def cdf(self, x: np.ndarray) -> np.ndarray:
#        pass
