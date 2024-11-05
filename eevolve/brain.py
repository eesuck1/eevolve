from typing import Iterable, Any, Callable

import torch
import numpy


class Brain:
    def __init__(self, mapping: Iterable[Any] | dict[float | int, Any] | Callable):
        self._mapping = mapping
        self._output = None
        self._decision = None

        self._model = torch.nn.Sequential()
        self._metric = torch.Tensor(0.0)

    def add_layer(self, layer: torch.nn.Module, name: str = None) -> None:
        if not isinstance(layer, torch.nn.Module):
            raise ValueError(f"Layer should be of `torch.nn.Module` type. {type(layer)} passed instead!")

        if name is None:
            name = f"{layer.__class__.__name__}_{len(self._model)}"

        self._model.add_module(name, layer)

    def add_layers(self, layers: Iterable[torch.nn.Module] | Any, names: Iterable[str] | Any = None) -> None:
        if names is None:
            names = [None for _ in range(len(layers))]
        elif len(layers) != len(names):
            raise ValueError(f"`layers` length must be equal `names` lengths."
                             f" {len(layers)} and {len(names)} passed instead!")

        for layer, name in zip(layers, names):
            self.add_layer(layer, name)

    def forward(self, observation: Iterable[Any] | Any, owner: Any, *args, **kwargs) -> None:
        self._output = self._model([float(value) for value in observation])

    def metric(self) -> None:
        pass

    def backward(self) -> None:
        pass

    def decide(self) -> Any:
        if self._mapping is None or not self._mapping:
            return None

        if isinstance(self._mapping, (list, tuple, numpy.ndarray)):
            return self._mapping[self._output]
        elif isinstance(self._mapping, dict):
            return self._mapping.get(self._output)
        elif callable(self._mapping):
            return self._mapping(self._output)
        else:
            raise ValueError(f"Mapping format is not supported `{type(self._mapping)}`")

    @property
    def decision(self) -> Any:
        return self._decision

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        self._model = model
