from typing import Iterable, Any, Callable

import numpy


class Brain:
    def __init__(self, mapping: Iterable[Any] | dict[float | int, Any] | Callable):
        self._mapping = mapping
        self._output = None
        self._decision = None

    def forward(self, observation: Iterable[Any] | Any, owner: Any, *args, **kwargs) -> None:
        self._output = None

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
