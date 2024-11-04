from typing import Any, Iterable
from eevolve.constants import _MAGNITUDE_EPSILON

import numpy


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
                 b: Iterable[float | int] | numpy.ndarray | Any) -> float:
        if len(a) == 2 and all((isinstance(x, (float, int)) for x in a)):
            x_1, y_1 = a
        else:
            x_1, y_1 = a.position

        if len(b) == 2 and all((isinstance(x, (float, int)) for x in b)):
            x_2, y_2 = b
        else:
            x_2, y_2 = b.position

        distance = numpy.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        return distance if distance > 0 else _MAGNITUDE_EPSILON
