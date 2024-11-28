from typing import Any, Callable

import numpy
import pygame

import eevolve


class WarAgent(eevolve.Agent):
    def __init__(self, agent_size: tuple[int | float, int | float],
                 agent_position: tuple[int | float, int | float] | numpy.ndarray, agent_name: str,
                 agent_surface: str | pygame.Surface | numpy.ndarray, brain: eevolve.Brain,
                 reproduce_threshold: float | int = 1, reproduce_count: int = 1, reproduce_function: Callable[["WarAgent"], "WarAgent"] = None):
        super().__init__(agent_size, agent_position, agent_name, agent_surface, brain, reproduce_threshold, reproduce_count, reproduce_function)

        self._health = eevolve.NumbersGenerator.uniform(offset=5.0, scaler=15.0)
        self._damage = eevolve.NumbersGenerator.uniform(offset=1.0)

    def win(self) -> None:
        self._health += eevolve.NumbersGenerator.uniform(scaler=5.0)
        self._damage += eevolve.NumbersGenerator.uniform()

    def attack(self, other: "WarAgent") -> None:
        if any((self.is_dead, other.is_dead)) or self.color == other.color:
            return

        self.stop()
        other.health -= self._damage

    @property
    def color(self) -> str:
        return self.name.split("_")[0]

    @property
    def health(self) -> float | Any:
        return self._health

    @health.setter
    def health(self, value: float) -> None:
        self._health = value

    @property
    def damage(self) -> float | Any:
        return self._damage

    @damage.setter
    def damage(self, value: float) -> None:
        self._damage = value
