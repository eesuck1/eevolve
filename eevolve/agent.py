from copy import deepcopy
from typing import Any, Sequence, Union, List, Callable

import numpy
import pygame

from eevolve.brain import Brain
from eevolve.loader import Loader
from eevolve.eemath import Math
from eevolve.constants import MAGNITUDE_EPSILON, COLLISION_UP, COLLISION_RIGHT, COLLISION_DOWN, COLLISION_LEFT


class Agent:
    def __init__(self, agent_size: tuple[int | float, int | float] | numpy.ndarray = (0, 0),
                 agent_position: tuple[int | float, int | float] | numpy.ndarray = (0, 0),
                 agent_name: str = "", agent_surface: str | pygame.Surface | numpy.ndarray = pygame.Surface((0, 0)),
                 brain: Brain = None, reproduce_threshold: float | int = 1, reproduce_count: int = 1, reproduce_function: Callable[["Agent"], "Agent"] = None):
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

        self._agent_surface = Loader.load_surface(agent_surface, agent_size)
        self._rect = pygame.FRect(agent_position, agent_size)
        self._sector_index = None

        self._brain = brain if brain is not None else Brain([])
        self._is_dead = False
        self._is_colliding_border = False
        self._colliding: Union["Agent", None, Any] = None
        self._colliding_directions = []

        self._reproduce_metric = 0
        self._reproduce_threshold = reproduce_threshold
        self._reproduce_count = reproduce_count
        self._reproduce_function = reproduce_function if reproduce_function is not None else self._default_reproduce
        self._children: list["Agent"] = []

        self._velocity = numpy.zeros((2,), dtype=float)

    def accelerate_by(self, delta: tuple[int | float, int | float] | numpy.ndarray) -> None:
        """
        Change Agent position by given delta within specified bounds with respect to current position.

        Example:

        agent.move_by((5, 5), (0, 0), game.display_size)

        :param delta: Delta X and Y which will be added to current Agent position.
        :return: None
        """
        delta = delta if isinstance(delta, numpy.ndarray) else numpy.array(delta)
        self._velocity += delta

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

    def accelerate_toward(self, point: Sequence[float | int] | numpy.ndarray | Any, value: float | int) -> None:
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

        self.accelerate_by((x * value, y * value))

    def move(self, delta_time: float, lower: tuple[int, int], upper: tuple[int, int], reset_velocity: bool = False) -> None:
        if upper[0] < self._rect.width or upper[0] < self._rect.height:
            raise ValueError(f"Upper bound minimum value is Agent size. {upper} given instead!")
        lower = (0, 0) if lower is None else lower
        upper = upper if upper is not None else (float('inf'), float('inf'))

        self._rect.x, collide_x = Math.clip(self._rect.x + self._velocity[0] * delta_time,
                                            lower[0], upper[0] - self._rect.width, return_bool=True)
        self._rect.y, collide_y = Math.clip(self._rect.y + self._velocity[1] * delta_time,
                                            lower[1], upper[1] - self._rect.height, return_bool=True)
        self._colliding_directions.clear()
        self._is_colliding_border = collide_x or collide_y

        if collide_x:
            if self._rect.x >= upper[0] - self._rect.width:
                self._colliding_directions.append(COLLISION_RIGHT)
            elif self._rect.x <= lower[0]:
                self._colliding_directions.append(COLLISION_LEFT)

        if collide_y:
            if self._rect.y >= upper[1] - self._rect.height:
                self._colliding_directions.append(COLLISION_DOWN)
            elif self._rect.y <= lower[1]:
                self._colliding_directions.append(COLLISION_UP)

        if reset_velocity:
            self._velocity = numpy.zeros(self._velocity.shape, dtype=float)

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

    def decide(self, observation: Sequence[Any], *args, **kwargs) -> Any:
        return self._brain(observation, self, *args, **kwargs)

    def die(self) -> None:
        self._is_dead = True

    def new_like_me(self) -> "Agent":
        new_agent = type(self)(self._agent_size, self.position, self._agent_name,
                               deepcopy(self._agent_surface), self._brain.new_like_me())

        for attribute, value in self.__dict__.items():
            if attribute.startswith("__"):
                continue

            setattr(new_agent, attribute, deepcopy(value))

        return new_agent

    def stop(self) -> None:
        self._velocity[0] = 0.0
        self._velocity[1] = 0.0

    def reproduce(self) -> None:
        if self._reproduce_metric < self._reproduce_threshold:
            return

        for index in range(self._reproduce_count):
            self._children.append(self._reproduce_function(self))

        self._reproduce_metric = 0.0

    @staticmethod
    def _default_reproduce(parent: Union["Agent", Any]) -> Union["Agent", Any]:
        child = deepcopy(parent)
        child.brain.mutate()
        child.name += "Child"

        return child

    @property
    def position(self) -> tuple[int | float, int | float]:
        return self._rect.topleft

    @property
    def rect(self) -> pygame.FRect:
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

    @size.setter
    def size(self, value: tuple[int | float, int | float]) -> None:
        self._rect.size = value
        self._agent_size = value

    @property
    def is_dead(self) -> bool:
        return self._is_dead

    @is_dead.setter
    def is_dead(self, value: bool) -> None:
        self._is_dead = value

    @property
    def sector_index(self) -> tuple[int, int] | None:
        return self._sector_index

    @sector_index.setter
    def sector_index(self, value: tuple[int, int]):
        self._sector_index = value

    @property
    def brain(self) -> Brain:
        return self._brain

    @property
    def velocity(self) -> numpy.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, value: numpy.ndarray) -> None:
        self._velocity = value

    @property
    def velocity_norm(self) -> float:
        return numpy.sqrt((self._velocity ** 2).sum())

    @property
    def colliding_border(self) -> bool:
        return self._is_colliding_border

    @colliding_border.setter
    def colliding_border(self, value: bool) -> None:
        self._is_colliding_border = value

    @property
    def collision_directions(self) -> list[Any]:
        return self._colliding_directions

    @collision_directions.setter
    def collision_directions(self, value: list[int, ...]) -> None:
        self._colliding_directions = value

    @property
    def colliding(self) -> Union["Agent", None, Any]:
        return self._colliding

    @colliding.setter
    def colliding(self, value: Union["Agent", None, Any]) -> None:
        self._colliding = value

    @property
    def reproduce_metric(self) -> float | int:
        return self._reproduce_metric

    @reproduce_metric.setter
    def reproduce_metric(self, value: float | int):
        self._reproduce_metric = value

    @property
    def children(self) -> list[Union["Agent", Any]]:
        return self._children

    def __str__(self) -> str:
        return f"<{self._agent_name}: ({self.position[0]}, {self.position[1]})>"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return 0
