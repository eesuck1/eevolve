import copy
from typing import Any, Callable, Iterable

import numpy
import pygame

from eevolve import Brain, Agent


class PositionGenerator:
    @staticmethod
    def uniform(game: Any, number: int) -> Iterable[numpy.ndarray]:
        """
        Generates a specified number of randomly generated uniformly distributed pairs of coordinates.

        Example:

        positions = eevolve.PositionGenerator.uniform(game, 10)

        for position in positions:
            print(position)

        :param game: `Game` class instance, needed to get display bounds.
        :param number: Number of coordinate pairs to generate.
        :return: Generator object that yields [x y] coordinates pair.
        """
        x = numpy.random.rand(number, 1) * game.display_size[0]
        y = numpy.random.rand(number, 1) * game.display_size[1]

        for pair in numpy.hstack((x, y)):
            yield pair

    @staticmethod
    def even(game: Any, number: int, offset_scaler: int = 10,
             lower: tuple[float | int, float | int] = None, upper: tuple[float | int, float | int] = None) -> Iterable[
        numpy.ndarray]:
        """
        Generates a specified number of evenly distributed coordinate pairs, considering an offset scaler.

        Example:

        positions = eevolve.PositionGenerator.even(game, 16)

        for position in positions:
            print(position)

        :param game: `Game` class instance, needed to get display bounds.
        :param number: Number of coordinate pairs to generate.
        :param offset_scaler: Used to offset a start and end point from the borders by `display_size / offset_scaler`.
        :param upper: Lower bound for X and Y, default value is (0, 0)
        :param lower: Upper bound for X and Y, default value is display size
        :return: Generator object that yields [x y] coordinates pair.
        """
        x_offset = game.display_size[0] / offset_scaler
        y_offset = game.display_size[1] / offset_scaler

        if lower is None:
            lower = (x_offset, y_offset)

        if upper is None:
            upper = (game.display_size[0] - x_offset, game.display_size[1] - y_offset)

        dividers = [1]

        for i in range(2, number):
            if number % i == 0:
                dividers.append(i)

        x, y = numpy.mgrid[lower[0]:upper[0]:dividers[len(dividers) // 2] * 1j,
               lower[1]:upper[1]:number / dividers[len(dividers) // 2] * 1j]

        for pair in numpy.column_stack((x.ravel(), y.ravel())):
            yield pair


class AgentGenerator:
    DEFAULT_SIZE_SCALER = 50
    DEFAULT_NAME = "DefaultAgent"
    DEFAULT_SURFACE_COLOR = (0, 0, 0)
    DEFAULT_BRAIN = Brain([])
    DEFAULT_POSITION = (0, 0)

    @staticmethod
    def default(game: Any, number: int,
                size: tuple[int | float, int | float] = None,
                surface: str | pygame.Surface | numpy.ndarray = None,
                position: tuple[int | float, int | float] = None,
                name_pattern: Callable[[int], str] = None,
                brain: Brain = None) -> Iterable[Agent]:
        """
        Generates a specified number of Agents with default or customized attributes.

        Example:

        agents = eevolve.AgentGenerator.default(game, 10, name_pattern=lambda index: f"Agent_{index ** 2}")

        for agent in agents:
            print(agent.name)           # Agent_0, Agent_1, Agent_4, ... Agent_81
            print(type(agent))          # Agent

        :param game: The `Game` class instance.
        :param number: The number of Agents to generate.
        :param size: The size of the Agents. Default is calculated as square width side length `display_width / 50`.
        :param surface: The surface for the Agents. Default is a black square with given `size`.
        :param position: The position of the Agents. Default is (0, 0).
        :param name_pattern: A function to generate Agent names with respect to iteration index.
        Default is a lambda function that increments an DefaultAgent name by 1: `DefaultAgent_1 -> DefaultAgent_2`.
        :param brain: The brain instance for the Agents. Default is `DEFAULT_BRAIN`.
        :return: Generator object that yields deepcopy of `default Agent`.
        """
        if size is None:
            scaler = AgentGenerator.DEFAULT_SIZE_SCALER
            size = (game.display_size[0] // scaler, game.display_size[0] // scaler)

        if surface is None:
            surface = numpy.full((*size, 3), AgentGenerator.DEFAULT_SURFACE_COLOR, dtype=numpy.uint8)

        if position is None:
            position = AgentGenerator.DEFAULT_POSITION

        if name_pattern is None:
            name_pattern = lambda i: f"{AgentGenerator.DEFAULT_NAME}_{i}"

        if brain is None:
            brain = AgentGenerator.DEFAULT_BRAIN

        agent = Agent(size, position, "", surface, brain)

        for index in range(number):
            agent.name = name_pattern(index)
            yield copy.deepcopy(agent)

    @staticmethod
    def like(agent: Agent, number: int, name_pattern: Callable[[int], str] = None) -> Iterable[Agent]:
        """
        Generates a specified number of agents as deepcopy of given 'base' Agent.
        :param agent: 'base' Agent instance
        :param number: number of 'base' Agent copies to generate
        :param name_pattern: A function to generate Agent names with respect to index.
        Default is a lambda function that increments an Agent name by 1: `Agent 1 -> Agent 2`.

        :return: Generator object that yields deepcopy of `default Agent`.

        Example:

        base_agent = eevolve.Agent(...)
        agents = eevolve.AgentGenerator.like(base_agent, 10, name_pattern=lambda index: f"Agent_{index ** 2}")

        for agent in agents:
            print(agent.name)           # Agent_0, Agent_1, Agent_4, ... Agent_81
            print(type(agent))          # Agent
            print(agent is base_agent)  # False
        """

        if name_pattern is None:
            name_pattern = lambda i: f"{AgentGenerator.DEFAULT_NAME}_{i}"

        for index in range(number):
            agent.name = name_pattern(index)
            yield copy.deepcopy(agent)


class ColorGenerator:
    @staticmethod
    def random(number: int, bounds: tuple[int, int] = None, return_tuple: bool = False) -> Any:
        if bounds is None:
            bounds = (0, 255)

        for _ in range(number):
            color = numpy.random.randint(low=bounds[0], high=bounds[1], size=(3,))

            yield tuple(color) if return_tuple else color
