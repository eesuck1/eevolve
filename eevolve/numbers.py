from typing import Any, Callable

import numpy


class NumbersGenerator:
    @staticmethod
    def uniform(shape: tuple[int, ...] = (), offset: float = 0.0, scaler: float = 1.0,
                dtype: Any = numpy.float64) -> numpy.ndarray | float:
        return dtype(scaler * numpy.random.rand(*shape) + offset)

    @staticmethod
    def uniform_generator(number: int, shape: tuple[int, ...] = (), offset: float = 0.0, scaler: float = 1.0,
                          dtype: Any = numpy.float64) -> Any:
        for _ in range(number):
            yield NumbersGenerator.uniform(shape, offset, scaler, dtype)

    @staticmethod
    def hypercube_generator(number: int, dimensions: int = 1, offset: float = 0.0, scaler: float = 1.0,
                            dtype: Any = numpy.float64) -> Any:
        for _ in range(number):
            yield numpy.full((dimensions,), NumbersGenerator.uniform(offset=offset, scaler=scaler), dtype=dtype)

    @staticmethod
    def normal(shape: tuple[int, ...] = (), offset: float = 0.0, scaler: float = 1.0) -> numpy.ndarray | float:
        return scaler * numpy.random.randn(*shape) + offset

    @staticmethod
    def normal_generator(number: int, shape: tuple[int, ...] = (), offset: float = 0.0, scaler: float = 1.0) -> Any:
        for _ in range(number):
            yield NumbersGenerator.normal(shape, offset, scaler)

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
