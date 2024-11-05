from copy import deepcopy
from typing import Any, Iterable

import numpy
import pygame

from eevolve.brain import Brain
from eevolve.loader import Loader
from eevolve.eemath import Math
from eevolve.constants import MAGNITUDE_EPSILON


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

        self._agent_surface = Loader.load_surface(agent_surface, agent_size)
        self._rect = pygame.Rect(agent_position, agent_size)

        self._brain = brain
        self._is_dead = False

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

        self._rect.x = Math.clip(self._rect.x + delta_x, lower[0], upper[0])
        self._rect.y = Math.clip(self._rect.y + delta_y, lower[1], upper[1])

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

    def move_toward(self, point: Iterable[float | int] | numpy.ndarray | Any, distance: float | int,
                    lower: tuple[int, int] = None, upper: tuple[int, int] = None) -> None:
        if isinstance(point, Agent):
            x_2, y_2 = point.position
        elif len(point) == 2 and all((isinstance(x, (float, int)) for x in point)):
            x_2, y_2 = point[0], point[1]
        else:
            raise ValueError(f"'point' instance should be 'Agent' of collection of two numbers, "
                             f"({', '.join((str(type(value)) for value in point))}) given instead!")

        x_1, y_1 = self.position

        magnitude = numpy.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        if magnitude <= MAGNITUDE_EPSILON:
            self.move_to((x_2, y_2))
            return

        x = (x_2 - x_1) / magnitude
        y = (y_2 - y_1) / magnitude

        lower = (0, 0) if lower is None else lower
        upper = upper if upper is not None else (float('inf'), float('inf'))

        self.move_by((x * distance, y * distance), lower, upper)

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

    def decide(self, observation: Iterable[Any], *args, **kwargs) -> Any:
        self._brain.forward(observation, self, *args, **kwargs)

        return self._brain.decide()

    def die(self) -> None:
        self._is_dead = True

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

    @property
    def size(self) -> tuple[int | float, int | float]:
        return self._rect.size

    @property
    def is_dead(self) -> bool:
        return self._is_dead

    @is_dead.setter
    def is_dead(self, value: bool) -> None:
        self._is_dead = value

    def __str__(self) -> str:
        return f"<{self._agent_name}: ({self.position[0]}, {self.position[1]})>"

    def __repr__(self) -> str:
        return str(self)

    def __copy__(self) -> "Agent":
        return type(self)(self._agent_size, self._agent_position,
                          self._agent_name, self._agent_surface, self._brain)

    def __deepcopy__(self, memodict) -> "Agent":
        new_agent = type(self)(self._agent_size, self._agent_position, self._agent_name,
                               deepcopy(self._agent_surface), deepcopy(self._brain))

        return new_agent

    def __len__(self) -> int:
        return 0
