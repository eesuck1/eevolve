import math
from itertools import combinations, chain
from typing import Any, Sequence, Callable

from eevolve.agent import Agent


class Board:
    def __init__(self, sector_size: tuple[int | float, int | float] = (0, 0), sectors_number: int = -1,
                 collision_timeout: Callable[[Agent | Any, Agent | Any], int] | int | float = None) -> None:
        self._sector_width, self._sector_height = sector_size
        self._sectors_number = sectors_number
        self._display_size = (self._sector_width * self._sectors_number - 1,
                              self._sector_height * self._sectors_number - 1)
        self._board: list[list[list[Agent]]] = [[[] for _ in range(sectors_number)] for _ in range(sectors_number)]
        self._collided: list[tuple[Agent, Agent]] = []
        self._collision_timer: dict[tuple[Agent, Agent], int] = {}
        self._sector_pairs: list[tuple[Agent, Agent]] = []
        self._dead_agents: list[Agent] = []
        self._agents: dict[Agent, list[Any]] = {}

        if collision_timeout is None:
            self._collision_timeout = lambda x, y: 250
        elif isinstance(collision_timeout, (int, float)):
            self._collision_timeout = lambda x, y: math.ceil(collision_timeout)
        elif isinstance(collision_timeout, Callable):
            self._collision_timeout = collision_timeout
        else:
            raise ValueError(f"`collision_timeout` must be function that takes [Agent, Agent] and return timeout "
                             f"or constant timeout value (int or float). {type(collision_timeout)} given instead!")

        self.__string = ""

    def add_agent(self, agent: Agent) -> None:
        if agent in self._agents:
            return
        x, y = agent.position
        x_i = math.floor(x / self._sector_width)
        y_i = math.floor(y / self._sector_height)

        agent.sector_index = (x_i, y_i)

        self._board[x_i][y_i].append(agent)
        self._agents[agent] = []

    def add_agents(self, agents: Sequence[Agent]) -> None:
        for agent in agents:
            self.add_agent(agent)

    def remove_agent(self, agent: Agent) -> None:
        if agent not in self.agents:
            return

        x_i, y_i = agent.sector_index

        self._board[x_i][y_i].remove(agent)
        self._agents.pop(agent, None)

    def move_agent(self, agent: Agent, delta_time: float) -> None:
        if agent not in self._agents:
            return

        agent.move(delta_time, (0, 0), self._display_size)

        x0_i, y0_i = agent.sector_index

        x1, y1 = agent.position

        x1_i = math.floor(x1 / self._sector_width)
        y1_i = math.floor(y1 / self._sector_height)

        if (x0_i, y0_i) != (x1_i, y1_i):
            self._board[x0_i][y0_i].remove(agent)
            self._board[x1_i][y1_i].append(agent)

            agent.sector_index = (x1_i, y1_i)

    def move_agents(self, delta_time: float) -> None:
        for agent in self._agents:
            self.move_agent(agent, delta_time)

    def check_collision(self) -> None:
        self._collided.clear()

        if len(self._agents) < 2:
            return

        for agent in self._agents:
            x0_i, y0_i = agent.sector_index

            x, y = agent.position
            width, height = agent.size

            indexes_to_check = [(x0_i, y0_i)]

            if x + width > x0_i * self._sector_width:
                x0_i = min(x0_i + 1, self._sectors_number - 1)
                indexes_to_check.append((x0_i, y0_i))
            if y + height > y0_i * self._sector_height:
                y0_i = min(y0_i + 1, self._sectors_number - 1)
                indexes_to_check.append((x0_i, y0_i))

            for other in filter(lambda other_agent: other_agent is not agent,
                                chain(*(self._board[i][j] for i, j in indexes_to_check))):
                if agent.is_collide(other):
                    if (self._collision_timer.get((agent, other), 0) == 0 or
                            self._collision_timer.get((other, agent), 0) == 0):
                        self._collided.append((agent, other))

                        timeout = self._collision_timeout(agent, other)

                        self._collision_timer[(agent, other)] = timeout
                        self._collision_timer[(other, agent)] = timeout

    def decrease_timeout(self, dt: int) -> None:
        for pair in self._collision_timer:
            self._collision_timer[pair] = max(self._collision_timer[pair] - dt, 0)

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

    def scan_around_agents(self, radius: int = 0, hold_previous: bool = False) -> None:
        if radius < 0:
            raise ValueError(f"Radius must be a non-negative integer. {radius} given instead!")

        for agent, other in self._agents.items():
            x_i, y_i = agent.sector_index

            if not hold_previous:
                other.clear()

            if radius:
                x_min = max(x_i - radius, 0)
                x_max = min(x_i + radius, self._sectors_number)
                y_min = max(y_i - radius, 0)
                y_max = min(y_i + radius, self._sectors_number)

                other.extend(chain(*[self._board[i][j]
                                    for i in range(x_min, x_max)
                                    for j in range(y_min, y_max)]))
            else:
                other.extend(self._board[x_i][y_i])

            if agent in other:
                other.remove(agent)

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
