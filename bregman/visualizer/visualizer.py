import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.base import BregmanObject, Coords, Point
from bregman.manifold.bisector import Bisector
from bregman.manifold.geodesic import Geodesic
from bregman.manifold.manifold import BregmanManifold


class NoPlottingRoutine(Exception):
    pass


class NoAnimationRoutine(Exception):
    pass


class BregmanObjectVisualizer(ABC):

    plot_list: list[tuple[BregmanObject, dict]] = []
    animate_list: list[tuple[BregmanObject, dict]] = []
    callback_list: list["VisualizerCallback"] = []

    def __init__(self, manifold: BregmanManifold) -> None:
        super().__init__()

        self.manifold = manifold

    def plot_object(self, obj: BregmanObject, **kwargs) -> None:
        self.plot_list.append((obj, kwargs))

    def animate_object(self, obj: BregmanObject, **kwargs) -> None:
        self.animate_list.append((obj, kwargs))

    def add_callback(self, callback: "VisualizerCallback") -> None:
        self.callback_list.append(callback)

    @abstractmethod
    def plot_point(self, coords: Coords, point: Point, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_geodesic(
        self, coords: Coords, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def plot_bisector(
        self, coords: Coords, bisector: Bisector, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def animate_geodesic(
        self, coords: Coords, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    def run_callbacks(
        self, obj: BregmanObject, coords: Coords, **kwarg
    ) -> None:

        for c in self.callback_list:
            c.call(obj, coords, self, **kwarg)

    def visualize(self, coords: Coords) -> None:
        for obj, kwarg in self.plot_list:
            match obj:
                case Point():
                    self.plot_point(coords, obj, **kwarg)
                case Geodesic():
                    self.plot_geodesic(coords, obj, **kwarg)
                case Bisector():
                    if self.manifold.dimension != 2:
                        warnings.warn(
                            f"Bisector {obj} cannot be visualized due to non-unique projection."
                        )
                    else:
                        self.plot_bisector(coords, obj, **kwarg)
                case _:
                    raise NoPlottingRoutine

        for obj, kwarg in self.animate_list:
            match obj:
                case Geodesic():
                    self.animate_geodesic(coords, obj, **kwarg)
                case _:
                    raise NoAnimationRoutine

        for obj, kwarg in self.plot_list:
            for c in self.callback_list:
                c.call(obj, coords, self, **kwarg)


TVisualizer = TypeVar("TVisualizer", bound=BregmanObjectVisualizer)


class VisualizerCallback(ABC, Generic[TVisualizer]):

    @abstractmethod
    def call(
        self,
        obj: BregmanObject,
        coords: Coords,
        visualizer: TVisualizer,
        **kwargs,
    ) -> None:
        pass
