import eevolve
import numpy
import pygame

from examples.fight.source.constants import AGENT_SIZE


class FightAgent(eevolve.Agent):
    def __init__(self, damage: float, health: float, agent_size: tuple[int | float, int | float] | numpy.ndarray = (0, 0),
                 agent_position: tuple[int | float, int | float] | numpy.ndarray = (0, 0),
                 agent_name: str = "", agent_surface: str | pygame.Surface | numpy.ndarray = pygame.Surface((0, 0)),
                 brain: eevolve.Brain = None):
        super().__init__(agent_size, agent_position, agent_name, agent_surface, brain)

        self._damage = damage
        self._health = health

    def attack(self, other: "FightAgent") -> None:
        self.stop()
        other.health -= self._damage

    @property
    def health(self) -> float:
        return self._health

    @health.setter
    def health(self, value: float) -> None:
        self._health = value

    def __str__(self) -> str:
        return f"<{self._agent_name} health: {self._health} damage: {self._damage}>"


class Simulation:
    WINDOW_SIZE = (1920, 1080)
    DISPLAY_SIZE = (192, 108)
    WINDOW_CAPTION = "Fight"
    WINDOW_BACKGROUND = "examples/fight/assets/bg.png"

    def __init__(self):
        self._game = eevolve.Game(self.DISPLAY_SIZE, self.WINDOW_SIZE, self.WINDOW_CAPTION, self.WINDOW_BACKGROUND, 1,
                  collision_timeout=100)

        self._red = FightAgent(eevolve.NumbersGenerator.uniform(), eevolve.NumbersGenerator.uniform(offset=2.0, scaler=10.0), AGENT_SIZE,
                               (self.DISPLAY_SIZE[0] // 5, (self.DISPLAY_SIZE[1] - AGENT_SIZE[1]) // 2),
                                  "Red", "examples/fight/assets/red.png")
        self._blue = FightAgent(eevolve.NumbersGenerator.uniform(), eevolve.NumbersGenerator.uniform(offset=2.0, scaler=10.0), AGENT_SIZE,
                                (self.DISPLAY_SIZE[0] * 4 // 5 - AGENT_SIZE[0], (self.DISPLAY_SIZE[1] - AGENT_SIZE[1]) // 2),
                                  "Blue", "examples/fight/assets/blue.png")


    @staticmethod
    def collision_handler(pair: tuple[FightAgent, FightAgent]) -> None:
        agent_1, agent_2 = pair

        agent_1.attack(agent_2)

        if agent_2.health <= 0.0:
            agent_2.die()
            return

        agent_2.attack(agent_1)

        if agent_1.health <= 0.0:
            agent_1.die()
            return

    def run(self) -> None:
        self._red.accelerate_by((25, 0))
        self._blue.accelerate_by((-25, 0))

        self._game.add_agent(self._red)
        self._game.add_agent(self._blue)

        collision_task = eevolve.CollisionTask(self.collision_handler, 0)

        self._game.add_task(collision_task)
        self._game.run()
