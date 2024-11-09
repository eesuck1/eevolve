from typing import Iterable, Any, Callable
from eevolve.layers import Layer

import numpy


class Brain:
    def __init__(self, mapping: Iterable[Any] | dict[float | int, Any] | Callable):
        self._mapping = mapping
        self._output = None
        self._decision = None

        self._layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        if not isinstance(layer, Layer):
            raise ValueError(f"`layer` must be `Layer` type or it subclass. {type(layer)} given instead!")

        self._layers.append(layer)

    def add_layers(self, layers: list[Layer]) -> None:
        for layer in layers:
            self.add_layer(layer)

    def forward(self, observation: Iterable[Any] | numpy.ndarray | Any, owner: Any = None,
                output_function: Callable[[numpy.ndarray], Any] = lambda x: x, *args, **kwargs) -> None:
        observation = numpy.array(observation, dtype=float)

        self._output = self._layers[0](observation)

        if len(self._layers) > 1:
            for layer in self._layers[1:]:
                self._output = layer(self._output)

        self._output = output_function(self._output)

    def decide(self) -> Any:
        if self._mapping is None or not self._mapping:
            return None

        if isinstance(self._mapping, (list, tuple, numpy.ndarray)):
            return self._mapping[int(self._output)]
        elif isinstance(self._mapping, dict):
            return self._mapping.get(self._output)
        elif callable(self._mapping):
            return self._mapping(self._output)
        else:
            raise ValueError(f"Mapping format is not supported `{type(self._mapping)}`")

    def mutate(self) -> None:
        for layer in self._layers:
            layer.mutate()

    def combine(self, another: "Brain") -> "Brain":
        if len(self._layers) != len(another.layers):
            raise ValueError(f"Lengths of the layers must match for both objects. "
                             f"{len(self._layers)} vs {len(another.layers)} instead!")

        new_brain = type(self)(self._mapping)

        for layer_1, layer_2 in zip(self._layers, another.layers):
            new_brain.add_layer(layer_1.combine(layer_2))

        return new_brain

    @property
    def decision(self) -> Any:
        return self._decision

    @property
    def output(self) -> Any:
        return self._output

    @property
    def layers(self) -> list[Layer, ...]:
        return self._layers

    def __call__(self, observation: Iterable[Any] | numpy.ndarray | Any, owner: Any = None,
                 output_function: Callable[[numpy.ndarray], Any] = lambda x: x, *args, **kwargs) -> Any:
        self.forward(observation, owner, output_function, *args, **kwargs)

        return self.decide()
