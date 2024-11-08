import os
import numpy
import eevolve

from examples.war.source.agent import WarAgent, WarBrain
from examples.war.source.constants import *


class Simulation:
    def __init__(self, field_size: int) -> None:
        self._game = eevolve.Game(DISPLAY_SIZE, SCREEN_SIZE, WINDOW_NAME,
                                  os.path.join(ASSETS_PATH, "bg.png"), SECTORS_NUMBER)

        self._field_size = field_size

    @staticmethod
    def movement_task_handler(board: eevolve.Board) -> None:
        for row in board.agents_board:
            for sector in row:
                for agent in sector:
                    board.move_agent(agent, agent.decide(sector))

    @staticmethod
    def collision_handler(board: eevolve.Board) -> None:
        for pair in board.collided:
            agent_1, agent_2 = pair

            if agent_1.color != agent_2.color:
                board.remove_agent(agent_1)
                board.remove_agent(agent_2)
            else:
                board.move_agent_toward(agent_1, agent_2, -5)
                board.move_agent_toward(agent_2, agent_1, -5)

    @staticmethod
    def brain_handler(output: int) -> tuple[int, int]:
        vertical = numpy.random.randint(-3, 4)

        if output == WarBrain.LEFT:
            return -3, vertical
        elif output == WarBrain.RIGHT:
            return 3, vertical

    def init_agents(self) -> None:
        agents_number = self._field_size // 2

        agent_size = (8, 8)
        agent_surface_blue = os.path.join(ASSETS_PATH, "blue.png")
        agent_surface_red = os.path.join(ASSETS_PATH, "red.png")

        agent_blue = WarAgent(agent_size, (0, 0), "Blue", agent_surface_blue,
                              WarBrain(self.brain_handler))
        agent_red = WarAgent(agent_size, (0, 0), "Red", agent_surface_red,
                             WarBrain(self.brain_handler))

        width, height = self._game.display_size

        self._game.add_agents(agents_number,
                              eevolve.AgentGenerator.like(
                                  agent_blue, agents_number, name_pattern=lambda index: f"Blue_{index}"),
                              eevolve.PositionGenerator.even(
                                  self._game, agents_number, upper=(width // 3, height * 9 // 10)))

        self._game.add_agents(agents_number,
                              eevolve.AgentGenerator.like(
                                  agent_red, agents_number, name_pattern=lambda index: f"Red_{index}"),
                              eevolve.PositionGenerator.even(
                                  self._game, agents_number, lower=(width // 1.5, height // 10)))

    def init_tasks(self) -> None:
        movement_task = eevolve.BoardTask(self.movement_task_handler, 50, priority=1)
        collision_task = eevolve.BoardTask(self.collision_handler, 0, priority=1)

        self._game.add_tasks((movement_task, collision_task))

    def run(self) -> None:
        self.init_agents()
        self.init_tasks()

        self._game.run()
