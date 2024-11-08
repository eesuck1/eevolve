import numpy


class Activation:
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return sample

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class Relu(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        sample[sample < 0.0] = 0.0

        return sample


class ParametricRelu(Activation):
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha

    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        sample[sample < 0.0] *= self._alpha

        return sample


class Tanh(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return numpy.tanh(sample)


class Sigmoid(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return 1.0 / (1.0 + numpy.exp(-sample))


class Softmax(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        corrected = sample - sample.max(axis=-1, keepdims=True)

        return numpy.exp(corrected) / numpy.exp(corrected).sum(axis=-1, keepdims=True)
