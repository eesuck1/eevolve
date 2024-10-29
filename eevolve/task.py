from typing import Callable, Any

from eevolve.agent import Agent
from eevolve.board import Board


class Task:
    def __init__(self, function: Callable[..., Any], period_ms: int, *args, **kwargs) -> None:
        self._function = function
        self._period_ms = period_ms
        self._timer = 0

        self._args = args
        self._kwargs = kwargs

    @property
    def period(self) -> float:
        return self._period_ms

    @property
    def timer(self) -> float:
        return self._timer

    @timer.setter
    def timer(self, value: float) -> None:
        self._timer = value

    def __call__(self, *args, **kwargs) -> Any:
        return self._function(*self._args, *args, **self._kwargs, **kwargs)


class CollisionTask(Task):
    def __init__(self, function: Callable[[tuple[Agent, Agent]], None], period_ms: int, *args, **kwargs) -> None:
        super().__init__(function, period_ms, *args, **kwargs)


class AgentTask(Task):
    def __init__(self, function: Callable[[Agent], None], period_ms: int, *args, **kwargs):
        super().__init__(function, period_ms, *args, **kwargs)


class FrameEndTask(Task):
    def __init__(self, function: Callable[[Any], None], *args, **kwargs) -> None:
        super().__init__(function, 0, *args, **kwargs)


class BoardTask(Task):
    def __init__(self, function: Callable[[Board], None], period_ms: int, *args, **kwargs):
        super().__init__(function, period_ms, *args, **kwargs)
