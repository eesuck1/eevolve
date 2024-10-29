from typing import Any

import numpy
import pygame

from eevolve.brain import Brain
from eevolve.utils import Utils


class Agent:
    def __init__(self, agent_size: tuple[int | float, int | float],
                 agent_position: tuple[int | float, int | float],
                 agent_name: str, agent_surface: str | pygame.Surface | numpy.ndarray,
                 brain: Brain):
        self._agent_size = agent_size
        self._agent_name = agent_name
        self._agent_position = agent_position

        self._agent_surface = Utils.load_surface(agent_surface, agent_size)
        self._rect = pygame.Rect(agent_position, agent_size)

        self._brain = brain

    def move_by(self, delta: tuple[int | float, int | float],
                lower: tuple[int, int], upper: tuple[int, int]) -> None:
        delta_x, delta_y = delta

        self._rect.x = Utils.clip(self._rect.x + delta_x, lower[0], upper[0])
        self._rect.y = Utils.clip(self._rect.y + delta_y, lower[1], upper[1])

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self._agent_surface, self.position)

    def is_collide(self, agent: Any) -> bool:
        return self._rect.colliderect(agent.rect)

    @property
    def position(self) -> tuple[int | float, int | float]:
        return self._rect.topleft

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def __str__(self) -> str:
        return f"<{self._agent_name}: ({self.position[0]}, {self.position[1]})>"

    def __repr__(self) -> str:
        return str(self)
