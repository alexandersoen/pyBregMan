import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.base import BregmanObject, Coords, Point
from bregman.manifold.bisector import Bisector
from bregman.manifold.geodesic import Geodesic
from bregman.manifold.manifold import BregmanManifold


class NoPlottingRoutine(Exception):
    """Exception for when an object is called to be plotted but no subroutine
    is defined for it.
    """

    pass


class NoAnimationRoutine(Exception):
    """Exception for when an object is called to be animated but no subroutine
    is defined for it.
    """

    pass


class BregmanVisualizer(ABC):
    """Abstract visualization class. The visualization class allows for
    different BregmanObjects to be registered for plotting or animation. Then
    when a visualization call is made, the visualizer class takes care of the
    back-end plotting procedure.

    The ordering of the visualization is the following: First objects are
    plotted. Then object animations are run. Finally, callbacks functions are
    evaluated.

    Parameters:
        manifold: Bregman manifold in which geometric objects are being plotted in.
        plot_list: Objects which are being plotted.
        animate_list: Objects which are being animated.
        callback_list: Visualizer callbacks for addition plotting behaviour.
    """

    def __init__(self, manifold: BregmanManifold) -> None:
        """Initialize visualizer.

        Args:
            manifold: Bregman manifold in which geometric objects are being plotted in.
        """
        super().__init__()

        self.manifold = manifold

        self.plot_list: list[tuple[BregmanObject, dict]] = []
        self.animate_list: list[tuple[BregmanObject, dict]] = []
        self.callback_list: list["VisualizerCallback"] = []

    def plot_object(self, obj: BregmanObject, **kwargs) -> None:
        """Add object to be plotted.

        Args:
            obj: Object to be plotted.
            **kwargs: Additional arguments to be passed into the plotting back-end.
        """
        self.plot_list.append((obj, kwargs))

    def animate_object(self, obj: BregmanObject, **kwargs) -> None:
        """Add object to be animated.

        Args:
            obj: Object to be plotted.
            **kwargs: Additional arguments to be passed into the animation back-end.
        """
        self.animate_list.append((obj, kwargs))

    def add_callback(self, callback: "VisualizerCallback") -> None:
        """Add callback function. These are run after plotting and animation
        routines.

        Args:
            callback: Callback function.
        """
        self.callback_list.append(callback)

    @abstractmethod
    def plot_point(self, coords: Coords, point: Point, **kwargs) -> None:
        """Back-end plotting function for Point objects.

        Args:
            coords: Coordinate for object to be plotted in.
            point: Point object to be plotted.
            **kwargs: Additional arguments used for plotting.
        """
        pass

    @abstractmethod
    def plot_geodesic(
        self, coords: Coords, geodesic: Geodesic, **kwargs
    ) -> None:
        """Back-end plotting function for Geodesic objects.

        Args:
            coords: Coordinate for object to be plotted in.
            geodesic: Geodesic object to be plotted.
            **kwargs: Additional arguments used for plotting.
        """
        pass

    @abstractmethod
    def plot_bisector(
        self, coords: Coords, bisector: Bisector, **kwargs
    ) -> None:
        """Back-end plotting function for Bisector objects.

        Args:
            coords: Coordinate for object to be plotted in.
            bisector: Bisector object to be plotted.
            **kwargs: Additional arguments used for plotting.
        """
        pass

    @abstractmethod
    def animate_geodesic(
        self, coords: Coords, geodesic: Geodesic, **kwargs
    ) -> None:
        """Back-end animation function for Geodesic objects.

        Args:
            coords: Coordinate for object to be animted in.
            geodesic: Geodesic object to be animated.
            **kwargs: Additional arguments used for animation.
        """
        pass

    def run_callbacks(
        self, obj: BregmanObject, coords: Coords, **kwarg
    ) -> None:
        """Run all callback function on specific object.

        Args:
            obj: Object callback functions are being evaluated on.
            coords: Coordinates the callback functions are being evaluated on.
            **kwarg: Additional arguments passed to callback function.
        """
        for c in self.callback_list:
            c.call(obj, coords, self, **kwarg)

    def visualize(self, coords: Coords) -> None:
        """Visualize all registered plots and animations.

        The ordering of the visualization is the following: First objects are
        plotted. Then object animations are run. Finally, callbacks functions are
        evaluated.

        Args:
            coords: Coordinate the visualization is being made in.
        """
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


TVisualizer = TypeVar("TVisualizer", bound=BregmanVisualizer)


class VisualizerCallback(ABC, Generic[TVisualizer]):
    """Visualizer callback function abstract class. Used to define callback
    functions in a visualizer.
    """

    @abstractmethod
    def call(
        self,
        obj: BregmanObject,
        coords: Coords,
        visualizer: TVisualizer,
        **kwargs,
    ) -> None:
        """Function call for the callback function.

        Args:
            obj: Object callback functions are being evaluated on.
            coords: Coordinates the callback functions are being evaluated on.
            visualizer: Visualizer the callback is being called in.
            **kwarg: Additional arguments passed to callback function.
        """
        pass
