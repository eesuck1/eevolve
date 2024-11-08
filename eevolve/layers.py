import numpy

from eevolve.activations import Activation
from eevolve.generator import NumbersGenerator


class Dense:
    def __init__(self, shape: tuple[int, int], activation: Activation = None, use_bias: bool = True) -> None:
        self._weights = NumbersGenerator.weights(shape)
        self._bias = NumbersGenerator.weights((1, shape[1])) \
            if use_bias \
            else numpy.zeros((1, shape[1]))
        self._activation = activation if activation is not None else Activation()

    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return self._activation(sample.dot(self._weights) + self._bias)


class Conv1D:
    def __init__(self) -> None:
        ...
