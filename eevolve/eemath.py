from typing import Any, Sequence
from eevolve.constants import MAGNITUDE_EPSILON

import numpy


class Distance:
    def __init__(self, value: float, from_to: tuple[Any, Any]) -> None:
        self._value = value
        self._from_to = from_to

    @property
    def value(self) -> float:
        return self._value

    @property
    def from_to(self) -> tuple[Any, Any]:
        return self._from_to

    def __float__(self) -> float:
        return float(self._value)

    def __str__(self) -> str:
        return f"<Distance: from {self._from_to[0]} to {self._from_to[1]} is {self._value}>"

    def __repr__(self) -> str:
        return str(self)


class Math:
    @staticmethod
    def clip(value: int | float | Any, a: int | float, b: int | float,
             return_bool: bool = False) -> int | float | tuple[int | float, bool]:
        """
        Clamps a given value between two bounds.

        Example 1:

        clamped_value = Loader.clip(10, 0, 5)

        :param return_bool:
        :param value:
            The value to be clamped.

        :param a:
            The lower bound.

        :param b:
            The upper bound.

        :return:
            The clamped value, which will be equal to `a` if the original value
            was less than `a`, equal to `b` if it was greater than `b`,
            or the original value if it lies within the bounds.
        """

        if a > b:
            raise ValueError(f"Lower bound must not be greater than upper. {a} vs {b} given instead!")

        if value < a:
            clamped_value, out_of_bounds = a, True
        elif value > b:
            clamped_value, out_of_bounds = b, True
        else:
            clamped_value, out_of_bounds = value, False

        return (clamped_value, out_of_bounds) if return_bool else clamped_value

    @staticmethod
    def distance(a: Sequence[float | int] | numpy.ndarray | Any,
                 b: Sequence[float | int] | numpy.ndarray | Any) -> float:
        if len(a) == 2 and all((isinstance(x, (float, int)) for x in a)):
            x_1, y_1 = a
        else:
            x_1, y_1 = a.position

        if len(b) == 2 and all((isinstance(x, (float, int)) for x in b)):
            x_2, y_2 = b
        else:
            x_2, y_2 = b.position

        distance = numpy.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        return distance if distance > 0 else MAGNITUDE_EPSILON

    @staticmethod
    def distances(a: Sequence[float | int] | numpy.ndarray | Any,
                  b: Sequence[Sequence[float | int] | numpy.ndarray | Any] | Any) -> numpy.ndarray:
        return numpy.array([Math.distance(a, other)
                            for other in b
                            if a is not other])

    @staticmethod
    def distances_float(a: Sequence[float | int] | numpy.ndarray | Any,
                        b: Sequence[Sequence[float | int] | numpy.ndarray | Any] | Any) -> numpy.ndarray:
        return numpy.array([Math.distance(a, other).value
                            for other in b
                            if a is not other])
