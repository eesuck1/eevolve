from typing import Iterable, Any, Callable

import numpy
import pygame

import eevolve


class WarBrain(eevolve.Brain):
    LEFT = 0
    RIGHT = 1

    def __init__(self, mapping: Iterable[Any] | dict[float | int, Any] | Callable):
        super().__init__(mapping)

    def forward(self, observation: Iterable[eevolve.Agent] | Any, owner: eevolve.Agent | Any = None, *args, **kwargs) -> None:
        count_allies = 0

        for agent in observation:
            if agent.color == owner.color:
                count_allies += 1

        if owner.color == "Blue":
            self._output = self.RIGHT if count_allies > len(observation) // 2 else self.LEFT
        else:
            self._output = self.LEFT if count_allies > len(observation) // 2 else self.RIGHT


class WarAgent(eevolve.Agent):
    _MAGNITUDE_EPSILON = 1

    def __init__(self, agent_size: tuple[int | float, int | float],
                 agent_position: tuple[int | float, int | float] | numpy.ndarray, agent_name: str,
                 agent_surface: str | pygame.Surface | numpy.ndarray, brain: eevolve.Brain):
        super().__init__(agent_size, agent_position, agent_name, agent_surface, brain)

    @property
    def color(self) -> str:
        return self.name.split("_")[0]
