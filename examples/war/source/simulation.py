import os

import numpy
from numpy.random.mtrand import Sequence

import eevolve
from eevolve import AroundAgentTask, CollisionTask, AgentTask, Task, BorderCollisionTask
from examples.war.source.war_agent import WarAgent


class Simulation:
    DISPLAY_SIZE = (1920, 1080)
    SCREEN_SIZE = (1920, 1080)
    SECTORS_NUMBER = 5
    AGENTS_TO_REPRODUCE = 2
    MAX_SPEED = 25.0

    WINDOW_NAME = "test"
    ASSETS_PATH = "examples/war/assets/"

    def __init__(self, field_size: int) -> None:
        self._game = eevolve.Game(self.DISPLAY_SIZE, self.SCREEN_SIZE, self.WINDOW_NAME, os.path.join(self.ASSETS_PATH, "bg.png"),
                                  self.SECTORS_NUMBER, collision_timeout=50)

        self._field_size = field_size
        self._mapping = numpy.array([[i * 10, j * 10] for i in range(-1, 2) for j in range(-1, 2)])

    @staticmethod
    def movement_handler(agent: WarAgent, others: Sequence[WarAgent], dt: float) -> None:
        observation = numpy.array([(other.color == agent.color) * 2 - 1 for other in others], dtype=float)

        if len(observation) < 16:
            observation = numpy.pad(observation, (0, 16 - len(observation)), "constant", constant_values=0.0)
        elif len(observation) > 16:
            observation = observation[:16]

        agent.accelerate_by(agent.decide(observation) * dt, (-Simulation.MAX_SPEED, Simulation.MAX_SPEED))

    @staticmethod
    def collision_handler(pair: tuple[WarAgent, WarAgent], _dt: float) -> None:
        agent_1, agent_2 = pair

        if agent_1.color == agent_2.color:
            return

        agent_1.attack(agent_2)

        if agent_2.health < 0:
            agent_1.win()
            agent_1.reproduce_metric += 1
            return

        agent_2.attack(agent_1)

        if agent_1.health < 0:
            agent_2.win()
            agent_2.reproduce_metric += 1
            return

    @staticmethod
    def check_reproduce_handler(agent: WarAgent, _dt: float) -> None:
        agent.reproduce()

    @staticmethod
    def age_handler(agent: WarAgent, dt: float) -> None:
        agent.health -= dt

    @staticmethod
    def die_handler(agent: WarAgent, _dt: float) -> None:
        if agent.health < 0.0:
            agent.die()

    @staticmethod
    def border_collision(agent: WarAgent) -> None:
        agent.stop()

    def init_agents_handler(self) -> None:
        agents_number = 5

        agent_size = (16, 16)
        agent_surface_blue = os.path.join(self.ASSETS_PATH, "blue.png")
        agent_surface_red = os.path.join(self.ASSETS_PATH, "red.png")

        brain = eevolve.Brain(self._mapping)
        brain.add_layers([
            eevolve.Dense((16, 8), activation=eevolve.Relu()),
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
        init_agents_task = Task(self.init_agents_handler, 1000)
        movement_task = AroundAgentTask(self.movement_handler, 100)
        collision_task = CollisionTask(self.collision_handler, 0)
        agent_task = AgentTask(self.check_reproduce_handler, 0)
        age_task = AgentTask(self.age_handler, 250)
        die_task = AgentTask(self.die_handler, 0)
        border_task = BorderCollisionTask(self.border_collision, 0)

        tasks = (init_agents_task, movement_task, collision_task, agent_task, age_task, die_task, border_task)
        self._game.add_tasks(tasks)

    def run(self) -> None:
        self.init_tasks()

        self._game.run()
