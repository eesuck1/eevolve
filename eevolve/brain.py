from typing import Iterable, Any, Sized


class Brain:
    def __init__(self, mapping: Iterable[Any] | dict[float | int, Any]):
        self._mapping = mapping
        self._output = 0

    def forward(self, observation: Iterable[float]) -> None:
        pass

    def decide(self) -> Any:
        if isinstance(self._mapping, Iterable[Any] & Sized):
            return self._mapping[self._output]
        elif isinstance(self._mapping, dict):
            return self._mapping.get(self._output)
        else:
            raise ValueError(f"Mapping format is not supported `{type(self._mapping)}`")
