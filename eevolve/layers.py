import copy

import numpy

from eevolve.activations import Activation
from eevolve.numbers import NumbersGenerator


class Layer:
    def __init__(self, shape: tuple[int, ...], activation: Activation = None, use_bias: bool = True,
                 mutation_scaler: float = 0.1) -> None:
        self._sigma = mutation_scaler
        self._shape = shape
        self._use_bias = use_bias

        self._weights = numpy.empty(shape)
        self._bias = numpy.empty(shape) if use_bias else numpy.empty(shape)
        self._activation = activation if activation is not None else Activation()

    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return sample

    def mutate(self) -> None:
        self._weights += NumbersGenerator.normal(self._shape, scaler=self._sigma)

    def combine(self, other: "Layer") -> "Layer":
        if self._shape != other.shape:
            raise ValueError(f"Weights shapes must be identical. {self._shape} vs {other.shape} given instead!")
        elif not isinstance(other, type(self)):
            raise ValueError(f"`self` and `other` types should match. {type(self)} vs {type(other)} given instead!")

        new_layer = type(self)(self._shape, self._activation, self._use_bias, self._sigma)

        weight_indexes = NumbersGenerator.indexes_split_like(self._weights, 2, strict=False)
        bias_indexes = NumbersGenerator.indexes_split_like(self._bias, 2, strict=False)

        new_layer.weights[weight_indexes[0]] = self._weights[weight_indexes[0]]
        new_layer.biases[bias_indexes[0]] = self._bias[bias_indexes[0]]

        new_layer.weights[weight_indexes[1]] = other.weights[weight_indexes[1]]
        new_layer.biases[bias_indexes[1]] = other.biases[bias_indexes[1]]

        return new_layer

    def new_like_me(self) -> "Layer":
        new_layer = type(self)(self._shape, self._activation, self._use_bias, self._sigma)

        new_layer.weights = numpy.empty(self._shape)
        new_layer.biases = numpy.empty(self._shape) if self._use_bias else numpy.empty(self._shape)

        return new_layer

    @property
    def weights(self) -> numpy.ndarray:
        return self._weights

    @weights.setter
    def weights(self, value: numpy.ndarray):
        self._weights = value

    @property
    def biases(self) -> numpy.ndarray:
        return self._bias

    @biases.setter
    def biases(self, value: numpy.ndarray):
        self._bias = value

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {self._shape}, {self._activation}>"

    def __repr__(self) -> str:
        return str(self)

    def __copy__(self) -> "Layer":
        new_layer = type(self)(self._shape, self._activation, self._use_bias, self._sigma)

        new_layer.weights = self._weights
        new_layer.biases = self._bias

        return new_layer

    def __deepcopy__(self, memodict: dict) -> "Layer":
        new_layer = type(self)(self._shape, self._activation, self._use_bias, self._sigma)

        new_layer.weights = copy.deepcopy(self._weights)
        new_layer.biases = copy.deepcopy(self._bias)

        return new_layer


class Dense(Layer):
    def __init__(self, shape: tuple[int, ...], activation: Activation = None, use_bias: bool = True,
                 mutation_scaler: float = 0.1):
        super().__init__(shape, activation, use_bias, mutation_scaler)

        self._weights = NumbersGenerator.weights(shape)
        self._bias = NumbersGenerator.weights((1, shape[1])) \
            if use_bias \
            else numpy.zeros((1, shape[1]))

    def new_like_me(self) -> "Layer":
        new_layer = type(self)(self._shape, self._activation, self._use_bias, self._sigma)

        new_layer.weights = NumbersGenerator.weights(self._shape)
        new_layer.biases = NumbersGenerator.weights((1, self._shape[1])) \
            if self._use_bias \
            else numpy.zeros((1, self._shape[1]))

        return new_layer

    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        if len(sample.shape) > 2:
            raise ValueError(f"Expected `shape` length for `sample` must be greater than 2. {len(sample.shape)} given instead.")
        elif len(sample.shape) == 1:
            sample = numpy.expand_dims(sample, axis=0)

        return self._activation(sample.dot(self._weights) + self._bias)


class Conv1D(Layer):
    def __init__(self, shape: tuple[int, ...], activation: Activation = None, use_bias: bool = True,
                 mutation_scaler: float = 0.1) -> None:
        super().__init__(shape, activation, use_bias, mutation_scaler)

        self._filters, self._kernel_size = shape

        self._weights = NumbersGenerator.weights((self._filters, self._kernel_size))
        self._bias = NumbersGenerator.weights((self._filters,)) \
            if self._use_bias \
            else numpy.zeros((self._filters,))

    def new_like_me(self) -> "Layer":
        new_layer = type(self)(self._shape, self._activation, self._use_bias, self._sigma)

        new_layer.weights = NumbersGenerator.weights((self._filters, self._kernel_size))
        new_layer.biases = NumbersGenerator.weights((self._filters,)) \
            if self._use_bias \
            else numpy.zeros((self._filters,))

        return new_layer

    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        if len(sample.shape) != 2:
            raise ValueError(f"Expected `shape` length for `sample` is 2. {len(sample.shape)} given instead!")
        elif sample.shape[0] != 1:
            raise ValueError(f"Expected first dimension of `sample` is 1. {sample.shape[0]} given instead!")

        result = numpy.zeros((self._filters, sample.shape[1] - self._kernel_size + 1))

        for index, (kernel, bias) in enumerate(zip(self._weights, self._bias)):
            result[index] = self._activation(numpy.convolve(sample[0], kernel, "valid") + bias)

        return result


class Argmax(Layer):
    def __init__(self, axis: int = -1, keepdims: bool = False, return_int: bool = False):
        super().__init__((0, 0))

        self._axis = axis
        self._keepdims = keepdims
        self._return_int = return_int

    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        if self._return_int:
            return numpy.argmax(sample, axis=self._axis)[0]
        else:
            return numpy.argmax(sample, axis=self._axis, keepdims=self._keepdims)

    def new_like_me(self) -> "Layer":
        return copy.copy(self)

    def __copy__(self) -> "Argmax":
        return type(self)(self._axis, self._keepdims, self._return_int)

    def __deepcopy__(self, memodict: dict) -> "Argmax":
        return copy.copy(self)
