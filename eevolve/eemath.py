from typing import Any, Iterable
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
        return self._value

    def __str__(self) -> str:
        return f"<Distance: from {self._from_to[0]} to {self._from_to[1]} is {self._value}>"

    def __repr__(self) -> str:
        return str(self)


class Math:
    @staticmethod
    def clip(value: int | float, a: int | float, b: int | float) -> int | float:
        """
        Clamps a given value between two bounds.

        Example 1:

        clamped_value = Loader.clip(10, 0, 5)

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

        value = a if value < a else value
        value = b if value > b else value

        return value

    @staticmethod
    def distance(a: Iterable[float | int] | numpy.ndarray | Any,
                 b: Iterable[float | int] | numpy.ndarray | Any) -> Distance:
        if len(a) == 2 and all((isinstance(x, (float, int)) for x in a)):
            x_1, y_1 = a
        else:
            x_1, y_1 = a.position

        if len(b) == 2 and all((isinstance(x, (float, int)) for x in b)):
            x_2, y_2 = b
        else:
            x_2, y_2 = b.position

        distance = numpy.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        return Distance(distance, (a, b)) if distance > 0 else Distance(MAGNITUDE_EPSILON, (a, b))

    @staticmethod
    def distances(a: Iterable[float | int] | numpy.ndarray | Any,
                  b: Iterable[Iterable[float | int] | numpy.ndarray | Any] | Any) -> numpy.ndarray:
        return numpy.array([Math.distance(a, other)
                            for other in b
                            if a is not other])

    @staticmethod
    def distances_float(a: Iterable[float | int] | numpy.ndarray | Any,
                        b: Iterable[Iterable[float | int] | numpy.ndarray | Any] | Any) -> numpy.ndarray:
        return numpy.array([Math.distance(a, other).value
                            for other in b
                            if a is not other])
