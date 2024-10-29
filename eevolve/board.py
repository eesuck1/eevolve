from itertools import combinations

from eevolve.agent import Agent


class Board:
    def __init__(self, sector_size: tuple[int | float, int | float] = (0, 0),
                 sectors_number: int = -1) -> None:
        self._sector_width, self._sector_height = sector_size
        self._sectors_number = sectors_number
        self._board = [[[] for _ in range(sectors_number)] for _ in range(sectors_number)]
        self._agents = set()
        self._collided: list[tuple[Agent, Agent], ...] = []
        self._sector_pairs: list[tuple[Agent, Agent], ...] = []

        self.__string = ""

    def add_agent(self, agent: Agent) -> None:
        if agent in self._agents:
            return
        x, y = agent.position

        self._board[x // self._sector_width][y // self._sector_height].append(agent)
        self._agents.add(agent)

    def remove_agent(self, agent: Agent) -> None:
        if agent not in self.agents:
            return

        x, y = agent.position

        x_i = x // self._sector_width
        y_i = y // self._sector_height

        self._board[x_i][y_i].remove(agent)
        self._agents.remove(agent)

    def move_agent(self, agent: Agent, delta: tuple[int | float, int | float]):
        x0, y0 = agent.position
        agent.move_by(delta, (0, 0),
                      (self._sectors_number * self._sector_width - 1, self._sectors_number * self._sector_height - 1))
        x1, y1 = agent.position

        x0_i = x0 // self._sector_width
        x1_i = x1 // self._sector_width

        y0_i = y0 // self._sector_height
        y1_i = y1 // self._sector_height

        if (x0_i != x1_i) or (y0_i != y1_i):
            self._board[x0_i][y0_i].remove(agent)
            self._board[x1_i][y1_i].append(agent)

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
    def collided(self) -> list[tuple[Agent, Agent], ...]:
        return self._collided

    @property
    def sector_pairs(self) -> list[tuple[Agent, Agent], ...]:
        return self._sector_pairs

    @property
    def agents_board(self) -> list[list[list[Agent]]]:
        return self._board

    @property
    def agents(self) -> set[Agent]:
        return self._agents
