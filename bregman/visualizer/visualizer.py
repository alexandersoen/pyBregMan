from abc import ABC, abstractmethod

from bregman.base import BregObject, CoordType, Point
from bregman.manifold.manifold import Geodesic, Manifold


class NoPlottingRoutine(Exception):
    pass


class NoAnimationRoutine(Exception):
    pass


class Visualizer(ABC):

    plot_list: list[tuple[BregObject, dict]] = []
    animate_list: list[tuple[BregObject, dict]] = []

    def __init__(self, manifold: Manifold) -> None:
        super().__init__()

        self.manifold = manifold

    def plot_object(self, obj: BregObject, **kwargs) -> None:
        self.plot_list.append((obj, kwargs))

    def animate_object(self, obj: BregObject, **kwargs) -> None:
        self.animate_list.append((obj, kwargs))

    @abstractmethod
    def plot_point(self, ctype: CoordType, point: Point, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_geodesic(
        self, ctype: CoordType, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def animate_geodesic(
        self, ctype: CoordType, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    def visualize(self, ctype: CoordType) -> None:
        for obj, kwarg in self.plot_list:
            match obj:
                case Point():
                    self.plot_point(ctype, obj, **kwarg)
                case Geodesic():
                    self.plot_geodesic(ctype, obj, **kwarg)
                case _:
                    raise NoPlottingRoutine

        for obj, kwarg in self.animate_list:
            match obj:
                case Geodesic():
                    self.animate_geodesic(ctype, obj, **kwarg)
                case _:
                    raise NoAnimationRoutine
