from typing import Any

import numpy


class NumbersGenerator:
    @staticmethod
    def uniform(shape: tuple[int, ...], offset: float = 0.0, scaler: float = 1.0) -> numpy.ndarray:
        return scaler * numpy.random.rand(*shape) + offset

    @staticmethod
    def normal(shape: tuple[int, ...], offset: float = 0.0, scaler: float = 1.0) -> numpy.ndarray:
        return scaler * numpy.random.randn(*shape) + offset

    @staticmethod
    def weights(shape: tuple[int, ...], offset: float = 0.0, scaler: float = 1.0) -> numpy.ndarray:
        return NumbersGenerator.uniform(shape, offset, scaler)

    @staticmethod
    def indexes_split(shape: tuple[int, ...], parts: int, strict: bool = True) -> tuple[
            tuple[numpy.ndarray | Any, ...], ...]:
        if strict and shape[-1] % parts != 0:
            raise ValueError("In `strict` mode, the `shape[-1]` must be evenly divisible by the number of parts!")

        indexes = numpy.arange(shape[-1])
        numpy.random.shuffle(indexes)

        return tuple([(Ellipsis, indexes[i::parts]) for i in range(parts)])

    @staticmethod
    def indexes_split_like(array: numpy.ndarray, parts: int, strict: bool = True) -> tuple[
            tuple[numpy.ndarray | Any, ...], ...]:
        if strict and array.shape[-1] % parts != 0:
            raise ValueError("In `strict` mode, the `shape[-1]` must be evenly divisible by the number of parts!")

        indexes = numpy.arange(array.shape[-1])
        numpy.random.shuffle(indexes)

        return tuple([(Ellipsis, indexes[i::parts]) for i in range(parts)])
