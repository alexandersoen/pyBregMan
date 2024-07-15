from bregman.base import CoordChange, Coords, Point


class UnrecordedCoordType(Exception):
    """Exception when unrecorded coordinate type is accessed."""

    pass


class UnrecordedCoordConversion(Exception):
    """Exception when specific coordinate conversion is not recorded."""

    pass


class Atlas:
    """The Atlas class aims to manage data coordinate conversion.
    In particular, it provides functionality of recording coordinate types,
    registering conversions between coordinates, and conversion of Point
    data to different coordinates.

    Parameters:
        dimension: Dimension of the canonical coordinates. Specifically, the size of the data.
        recorded_coords: Set of coordinates which have been registered.
        transitions: Set of transitions (coordinate changes) which have been registered.
    """

    def __init__(self, dimension: int) -> None:
        """Initializes Atlas given the dimension of the data.

        Args:
            dimension: Dimension of the data being managed.
        """
        super().__init__()

        self.dimension = dimension

        self.recorded_coords: set[Coords] = set()
        self.transitions: dict[tuple[Coords, Coords], CoordChange] = dict()

    def add_coords(self, coords: Coords) -> None:
        """Adds coordinate to Atlas.

        Args:
            coords: Coordinate type to add to Atlas.
        """
        self.recorded_coords.add(coords)

    def add_transition(
        self,
        source_coords: Coords,
        target_coords: Coords,
        transition: CoordChange,
    ) -> None:
        """
        Adds transition function (coordinate transform) to Atlas.

        Args:
            source_coords: Source coordinates for the coordinate transform.
            target_coords: Target coordinates for the coordinate transform.
            transition: Function which changes data from source to target coordinates.
        """
        self.add_coords(source_coords)
        self.add_coords(target_coords)

        self.transitions[(source_coords, target_coords)] = transition

    def convert_coord(self, target_coords: Coords, point: Point) -> Point:
        r"""Converts coordinates of Point objects.

        Args:
            target_coords: Coords which one wants to convert point to.
            point: Point object being converted.

        Returns:
            point converted to target_coords coordinates based on manifold.
        """
        source_coord = point.coords

        if (
            source_coord not in self.recorded_coords
            or target_coords not in self.recorded_coords
        ):
            raise UnrecordedCoordType

        if source_coord == target_coords:
            return point

        try:
            coord_change = self.transitions[(source_coord, target_coords)]
        except KeyError:
            raise UnrecordedCoordConversion

        new_data = coord_change(point.data)

        return Point(coords=target_coords, data=new_data)

    def __call__(self, target_coords: Coords, point: Point) -> Point:
        r"""Converts coordinates of Point objects.

        Args:
            target_coords: Coords which one wants to convert point to.
            point: Point object being converted.

        Returns:
            point converted to target_coords coordinates based on manifold.
        """
        return self.convert_coord(target_coords, point)
