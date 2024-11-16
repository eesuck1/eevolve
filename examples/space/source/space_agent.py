import copy
from typing import Any

import pygame
import numpy
import eevolve


class SpaceAgent(eevolve.Agent):
    def __init__(self, mass: float, agent_size: tuple[int | float, int | float] | numpy.ndarray = (0, 0),
                 agent_position: tuple[int | float, int | float] | numpy.ndarray = (0, 0),
                 agent_name: str = "", agent_surface: str | pygame.Surface | numpy.ndarray = pygame.Surface((0, 0)),
                 brain: eevolve.Brain = None) -> None:
        super().__init__(agent_size, agent_position, agent_name, agent_surface, brain)

        self._mass = mass
        self._velocity = eevolve.NumbersGenerator.uniform((2,), scaler=50.0)

    def new_like_me(self) -> "SpaceAgent":
        new_agent = type(self)(self._mass, self._agent_size, self.position, self._agent_name,
                               copy.deepcopy(self._agent_surface), self._brain.new_like_me())

        for attribute, value in self.__dict__.items():
            if attribute.startswith("__"):
                continue

            if isinstance(value, (list, numpy.ndarray)):
                setattr(new_agent, attribute, copy.deepcopy(value))

        return new_agent

    def is_collide(self, agent: Any) -> bool:
        return eevolve.Math.distance(self._rect.center, agent.rect.center) <= (self._rect.width / 2 + agent.rect.width / 2)

    @property
    def mass(self) -> float:
        return numpy.pi * self._rect.width ** 2 / 4

    @mass.setter
    def mass(self, value: float) -> None:
        self._mass = value
