import copy
import os

from typing import Any

import numpy
import eevolve

from examples.war.source.agent import WarAgent


class Simulation:
    DISPLAY_SIZE = (256, 144)
    SCREEN_SIZE = (1280, 720)
    SECTORS_NUMBER = 5
    AGENTS_TO_REPRODUCE = 2

    WINDOW_NAME = "test"
    ASSETS_PATH = "examples/war/assets/"

    def __init__(self, field_size: int) -> None:
        self._game = eevolve.Game(self.DISPLAY_SIZE, self.SCREEN_SIZE, self.WINDOW_NAME, os.path.join(self.ASSETS_PATH, "bg.png"),
                                  self.SECTORS_NUMBER)

        self._field_size = field_size
        self._mapping = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

    def init_agents(self) -> None:
        agents_number = self._field_size // 2

        agent_size = (4, 4)
        agent_surface_blue = os.path.join(self.ASSETS_PATH, "blue.png")
        agent_surface_red = os.path.join(self.ASSETS_PATH, "red.png")

        brain = eevolve.Brain(self._mapping)
        brain.add_layers([
            eevolve.Dense((10, 8), activation=eevolve.Relu()),
            eevolve.Dense((8, 8), activation=eevolve.Relu()),
            eevolve.Dense((8, len(self._mapping)), activation=eevolve.Softmax()),
            eevolve.Argmax(return_int=True),
        ])

        agent_blue = WarAgent(agent_size, (0, 0), "Blue", agent_surface_blue, brain)
        agent_red = WarAgent(agent_size, (0, 0), "Red", agent_surface_red, brain)

        self._game.add_agents(agents_number,
                              eevolve.AgentGenerator.like(
                                  agent_blue, agents_number, name_pattern=lambda index: f"Blue_{index}"))

        self._game.add_agents(agents_number,
                              eevolve.AgentGenerator.like(
                                  agent_red, agents_number, name_pattern=lambda index: f"Red_{index}"))

    def init_tasks(self) -> None:
        # TODO: add tasks
        pass

    def run(self) -> None:
        self.init_agents()
        self.init_tasks()

        self._game.run()
