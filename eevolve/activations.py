import numpy


class Activation:
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return sample


class Relu(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        sample[sample < 0.0] = 0.0

        return sample


class Tanh(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return numpy.tanh(sample)


class Sigmoid(Activation):
    def __call__(self, sample: numpy.ndarray) -> numpy.ndarray:
        return 1.0 / (1.0 + numpy.exp(-sample))
