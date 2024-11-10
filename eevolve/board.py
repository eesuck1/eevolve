from itertools import combinations
from typing import Iterable, Any

import numpy

from eevolve.agent import Agent
from eevolve.eemath import Math


class Board:
    def __init__(self, sector_size: tuple[int | float, int | float] = (0, 0),
                 sectors_number: int = -1) -> None:
        self._sector_width, self._sector_height = sector_size
        self._sectors_number = sectors_number
        self._display_size = (self._sector_width * self._sectors_number - 1,
                              self._sector_height * self._sectors_number - 1)
        self._board: list[list[list[Agent]]] = [[[] for _ in range(sectors_number)] for _ in range(sectors_number)]
        self._collided: list[tuple[Agent, Agent]] = []
        self._sector_pairs: list[tuple[Agent, Agent]] = []
        self._dead_agents: list[Agent] = []
        self._agents: dict[Agent, list[Any]] = {}

        self.__string = ""

    def add_agent(self, agent: Agent) -> None:
        if agent in self._agents:
            return
        x, y = agent.position
        x_i = x // self._sector_width
        y_i = y // self._sector_height

        agent.sector_index = (x_i, y_i)

        self._board[x_i][y_i].append(agent)
        self._agents[agent] = []

    def remove_agent(self, agent: Agent) -> None:
        if agent not in self.agents:
            return

        x_i, y_i = agent.sector_index

        self._board[x_i][y_i].remove(agent)
        self._agents.pop(agent, None)

    def move_agent(self, agent: Agent, delta: tuple[int | float, int | float] | Iterable[float | int]) -> None:
        if agent not in self._agents:
            return

        upper_x, upper_y = self._display_size
        agent_width, agent_height = agent.size

        agent.move_by(delta, (0, 0), (upper_x - agent_width, upper_y - agent_height))
        x1, y1 = agent.position

        x0_i, y0_i = agent.sector_index

        x1_i = x1 // self._sector_width
        y1_i = y1 // self._sector_height

        if (x0_i, y0_i) != (x1_i, y1_i):
            self._board[x0_i][y0_i].remove(agent)
            self._board[x1_i][y1_i].append(agent)

            agent.sector_index = (x1_i, y1_i)

    def move_agent_toward(self, agent: Agent, point: Iterable[float | int] | numpy.ndarray | Any,
                          distance: float | int) -> None:
        if agent not in self._agents:
            return

        upper_x, upper_y = self._display_size
        agent_width, agent_height = agent.size

        agent.move_toward(point, distance, (0, 0), (upper_x - agent_width, upper_y - agent_height))
        x1, y1 = agent.position

        x0_i, y0_i = agent.sector_index

        x1_i = x1 // self._sector_width
        y1_i = y1 // self._sector_height

        if (x0_i, y0_i) != (x1_i, y1_i):
            self._board[x0_i][y0_i].remove(agent)
            self._board[x1_i][y1_i].append(agent)

            agent.sector_index = (x1_i, y1_i)

    def check_collision(self) -> None:
        self._collided.clear()

        if len(self._agents) < 2:
            return

        for row in self._board:
            for sector in row:
                if len(sector) < 2:
                    continue

                for pair in combinations(sector, 2):
                    first, second = pair

                    if first.is_collide(second):
                        self._collided.append((first, second))

    def check_sector_pairs(self) -> None:
        self._sector_pairs.clear()

        if len(self._agents) < 2:
            return

        for row in self._board:
            for sector in row:
                if len(sector) < 2:
                    continue

                for pair in combinations(sector, 2):
                    self._sector_pairs.append(pair)

    def check_dead(self) -> None:
        self._dead_agents.clear()

        for agent in self._agents:
            if agent.is_dead:
                self._dead_agents.append(agent)

    def scan_around_agent(self, agent: Agent, radius: int = 0, hold_previous: bool = False) -> None:
        if radius < 0:
            raise ValueError(f"Radius must be a non-negative integer. {radius} given instead!")
        if agent not in self._agents:
            return

        x_i, y_i = agent.sector_index

        agents = self._agents[agent]

        if not hold_previous:
            agents.clear()

        if radius:
            x_min = max(x_i - radius, 0)
            x_max = min(x_i + radius, self._sectors_number)
            y_min = max(y_i - radius, 0)
            y_max = min(y_i + radius, self._sectors_number)

            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    agents.extend(self._board[i][j])
        else:
            agents.extend(self._board[x_i][y_i])

        if agent in agents:
            agents.remove(agent)

    def scan_distances_around_agent(self, agent: Agent, radius: int = 0, hold_previous: bool = False) -> None:
        if radius < 0:
            raise ValueError(f"Radius must be a non-negative integer. {radius} given instead!")
        if agent not in self._agents:
            return

        x_i, y_i = agent.sector_index

        if radius:
            x_min = max(x_i - radius, 0)
            x_max = min(x_i + radius, self._sectors_number)
            y_min = max(y_i - radius, 0)
            y_max = min(y_i + radius, self._sectors_number)

            distances = [
                Math.distance(agent, other)
                for i in range(x_min, x_max)
                for j in range(y_min, y_max)
                for other in self._board[i][j]
                if other is not agent
            ]
        else:
            distances = [Math.distance(agent, other)
                         for other in self._board[x_i][y_i]
                         if other is not agent]

        agents = self._agents[agent]

        if not hold_previous:
            agents.clear()
        agents.extend(distances)

    def __str__(self) -> str:
        self.__string = ""
        self.__string += "-" * 128 + "\n"
        self.__string += " " * 57 + "<Board>\n"

        for i in range(self._sectors_number):
            self.__string += "-" * 128 + "\n"
            for j in range(self._sectors_number):
                self.__string += f"[{i}, {j}]: {', '.join([str(agent) for agent in self._board[i][j]])}\n"

        self.__string += "-" * 128 + "\n"
        return self.__string

    @property
    def sectors_number(self) -> int:
        return self._sectors_number

    @property
    def sector_size(self) -> tuple[int, int]:
        return self._sector_width, self._sector_height

    @property
    def collided(self) -> list[tuple[Agent | Any, Agent | Any]]:
        return self._collided

    @property
    def sector_pairs(self) -> list[tuple[Agent | Any, Agent | Any]]:
        return self._sector_pairs

    @property
    def agents_board(self) -> list[list[list[Agent | Any]]]:
        return self._board

    @property
    def agents(self) -> dict[Agent | Any, list[Any]]:
        return self._agents

    @property
    def dead(self) -> list[Agent | Any]:
        return self._dead_agents
