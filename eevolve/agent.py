from typing import Any
from copy import deepcopy

import numpy
import pygame

from eevolve.brain import Brain
from eevolve.utils import Utils


class Agent:
    def __init__(self, agent_size: tuple[int | float, int | float],
                 agent_position: tuple[int | float, int | float] | numpy.ndarray,
                 agent_name: str, agent_surface: str | pygame.Surface | numpy.ndarray,
                 brain: Brain):
        """
        Initializes a new agent.

        :param agent_size: The size of the Agent surface.
        :param agent_position: The initial position of the Agent.
        :param agent_name: The name of the agent.
        :param agent_surface: If passes string image by this path will be loaded,
        numpy bitmap array will be converted to pygame.Surface, pygame.Surface will be loaded directly.
        :param brain: The Brain class instance for the Agent.

        Example:
            brain_instance = Brain(...)

            agent = Agent((50, 50), (100, 100), "Agent_1", "path/to/image.png", brain_instance)
        """
        self._agent_size = agent_size
        self._agent_name = agent_name
        self._agent_position = agent_position

        self._agent_surface = Utils.load_surface(agent_surface, agent_size)
        self._rect = pygame.Rect(agent_position, agent_size)

        self._brain = brain

    def move_by(self, delta: tuple[int | float, int | float],
                lower: tuple[int, int], upper: tuple[int, int]) -> None:
        """
        Change Agent position by given delta within specified bounds with respect to current position.

        Example:

        agent.move_by((5, 5), (0, 0), game.display_size)

        :param delta: Delta X and Y which will be added to current Agent position.
        :param lower: Lower bound for X and Y.
        :param upper: Upper bound for X and Y.
        :return: None
        """
        delta_x, delta_y = delta

        self._rect.x = Utils.clip(self._rect.x + delta_x, lower[0], upper[0])
        self._rect.y = Utils.clip(self._rect.y + delta_y, lower[1], upper[1])

    def move_to(self, position: tuple[int | float, int | float] | numpy.ndarray) -> None:
        """
        Set Agent position to given value.

        Example:

        agent.move_to((100, 200))

        :param position: New X and Y coordinate of Agent.
        :return: None
        """
        self._rect.x = position[0]
        self._rect.y = position[1]

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draws the agent on a given surface.

        Example:
            screen = pygame.display.set_mode((800, 600))
            agent.draw(screen)

        :param surface: The Surface on which to draw an Agent Surface.
        :return: None
        """
        surface.blit(self._agent_surface, self.position)

    def is_collide(self, agent: Any) -> bool:
        """
        Checks if the Agent collides with another Agent.

        Example:

        if agent.is_collide(other_agent):
            print("Collision detected!")

        :param agent: Agent instance to check collision with.
        :return: True if the agents collide, False otherwise.
        """
        return self._rect.colliderect(agent.rect)

    @property
    def position(self) -> tuple[int | float, int | float]:
        return self._rect.topleft

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    @property
    def name(self) -> str:
        return self._agent_name

    @name.setter
    def name(self, value: str):
        self._agent_name = value

    def __str__(self) -> str:
        return f"<{self._agent_name}: ({self.position[0]}, {self.position[1]})>"

    def __repr__(self) -> str:
        return str(self)

    def __copy__(self) -> "Agent":
        return Agent(self._agent_size, self._agent_position,
                     self._agent_name, self._agent_surface, self._brain)

    def __deepcopy__(self, memodict) -> "Agent":
        new_agent = Agent(self._agent_size, self._agent_position, self._agent_name,
                          deepcopy(self._agent_surface), deepcopy(self._brain))

        return new_agent
