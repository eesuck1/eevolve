import copy
from typing import Any, Callable, Iterable

import numpy
import pygame

from eevolve import Brain, Agent


class PositionGenerator:
    @staticmethod
    def uniform(game: Any, number: int) -> Iterable[numpy.ndarray]:
        x = numpy.random.rand(number, 1) * game.display_size[0]
        y = numpy.random.rand(number, 1) * game.display_size[1]

        for pair in numpy.hstack((x, y)):
            yield pair

    @staticmethod
    def even(game: Any, number: int, offset_scaler: int = 10) -> Iterable[numpy.ndarray]:
        x_offset = game.display_size[0] / offset_scaler
        y_offset = game.display_size[1] / offset_scaler

        dividers = [1]

        for i in range(2, number):
            if number % i == 0:
                dividers.append(i)

        x, y = numpy.mgrid[x_offset:game.display_size[0] - x_offset:dividers[len(dividers) // 2] * 1j,
                           y_offset:game.display_size[1] - y_offset:number / dividers[len(dividers) // 2] * 1j]

        for pair in numpy.column_stack((x.ravel(), y.ravel())):
            yield pair


class AgentGenerator:
    DEFAULT_SIZE_SCALER = 50
    DEFAULT_NAME = "Default Agent 0"
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
        if size is None:
            scaler = AgentGenerator.DEFAULT_SIZE_SCALER
            size = (game.display_size[0] // scaler, game.display_size[0] // scaler)

        if surface is None:
            surface = numpy.full((*size, 3), AgentGenerator.DEFAULT_SURFACE_COLOR, dtype=numpy.uint8)

        if position is None:
            position = AgentGenerator.DEFAULT_POSITION

        if name_pattern is None:
            name_pattern = lambda i: AgentGenerator.DEFAULT_NAME[:-1] + str(i)

        if brain is None:
            brain = AgentGenerator.DEFAULT_BRAIN

        agent = Agent(size, position, "", surface, brain)

        for index in range(number):
            agent.name = name_pattern(index)
            yield copy.deepcopy(agent)

    @staticmethod
    def like(agent: Agent, number: int, name_pattern: Callable[[int], str] = None) -> Iterable[Agent]:
        if name_pattern is None:
            name_pattern = lambda i: AgentGenerator.DEFAULT_NAME[:-1] + str(i)

        for index in range(number):
            agent.name = name_pattern(index)
            yield copy.deepcopy(agent)
