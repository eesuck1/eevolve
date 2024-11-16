from typing import Any

import numpy
import pygame

import eevolve


class WarAgent(eevolve.Agent):
    def __init__(self, agent_size: tuple[int | float, int | float],
                 agent_position: tuple[int | float, int | float] | numpy.ndarray, agent_name: str,
                 agent_surface: str | pygame.Surface | numpy.ndarray, brain: eevolve.Brain):
        super().__init__(agent_size, agent_position, agent_name, agent_surface, brain)

        self._health = eevolve.NumbersGenerator.normal(offset=10.0, scaler=100.0)
        self._damage = eevolve.NumbersGenerator.normal(offset=10.0, scaler=10.0)

        self._wins_counter = 0
        self._wins_threshold = 1
        self._can_reproduce = False
        self._is_fighting = False

        self._fight_surface = pygame.transform.scale_by(self._agent_surface, 1.5)
        self._default_surface = self._agent_surface

    def win(self) -> None:
        self._health += 20.0
        self._damage += 1.0
        self._wins_counter += 1
        self._is_fighting = False
        self._agent_surface = self._default_surface

        if self._wins_counter > self._wins_threshold:
            self._can_reproduce = True

    def reproduced(self) -> None:
        self._wins_counter = 0
        self._can_reproduce = False

    def fight(self, other: "WarAgent") -> None:
        if any((self.is_dead, other.is_dead)) or self.color == other.color:
            self._is_fighting = False
            other.is_fighting = False

            return

        self._is_fighting = True
        other.is_fighting = True

        self._agent_surface = self._fight_surface
        other.surface = self._fight_surface

        other.health -= self._damage
        self._health -= other.damage

        if self._health <= 0:
            self.die()
            other.win()
        if other.health <= 0:
            other.die()
            self.win()

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

    @property
    def can_reproduce(self) -> bool:
        return self._can_reproduce

    @property
    def is_fighting(self) -> bool:
        return self._is_fighting

    @is_fighting.setter
    def is_fighting(self, value: bool) -> None:
        self._is_fighting = value

    @property
    def surface(self) -> pygame.Surface:
        return self._agent_surface

    @surface.setter
    def surface(self, value: pygame.Surface) -> None:
        self._agent_surface = value
