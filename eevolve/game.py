import sys

import numpy
import pygame

from typing import Iterable
from eevolve.agent import Agent
from eevolve.task import Task, FrameEndTask, CollisionTask, AgentTask, BoardTask
from eevolve.utils import Utils
from eevolve.board import Board


class Game:
    _TOP_LEFT = (0, 0)

    def __init__(self,
                 display_size: tuple[float | int, float | int],
                 screen_size: tuple[float | int, float | int],
                 window_caption: str,
                 display_background: str | pygame.Surface | numpy.ndarray,
                 board_sectors_number: int,
                 agents_list: Iterable[Agent] = None,
                 tasks_list: Iterable[Task] = None):

        self._display = pygame.Surface(display_size)
        self._screen = pygame.display.set_mode(screen_size)
        self._clock = pygame.Clock()

        self._display_size = display_size
        self._screen_size = screen_size
        self._window_caption = window_caption

        self._agents_list = agents_list if agents_list is not None else []
        self._tasks = tasks_list if tasks_list is not None else []
        self._delta_time = 0

        self._background = Utils.load_surface(display_background, display_size)
        self._board = Board(
            (self.display_size[0] // board_sectors_number,
             self.display_size[1] // board_sectors_number),
            board_sectors_number)

        for agent in self._agents_list:
            self._board.add_agent(agent)

    def _init_internal_tasks(self) -> None:
        self.add_task(FrameEndTask(lambda: self._board.check_collision()))
        self.add_task(FrameEndTask(lambda: self.draw()))
        self.add_task(FrameEndTask(lambda: self._board.check_sector_pairs()))

    def draw(self) -> None:
        self._display.blit(self._background)
        agents = self._board.agents

        for agent in agents:
            agent.draw(self._display)

    def do_tasks(self) -> None:
        for task in self._tasks:
            task.timer += self._delta_time

            if isinstance(task, CollisionTask) and task.timer >= task.period:
                for collision_pair in self._board.collided:
                    task(collision_pair)
            elif isinstance(task, AgentTask) and task.timer >= task.period:
                for agent in self._board.agents:
                    task(agent)
            elif isinstance(task, BoardTask) and task.timer >= task.period:
                task(self._board)
            elif isinstance(task, FrameEndTask):
                task()
            elif task.timer >= task.period:
                task()
            else:
                continue

            task.timer = 0

    def run(self) -> None:
        self._init_internal_tasks()

        pygame.display.set_caption(self._window_caption)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self._delta_time = self._clock.get_time()
            self.do_tasks()

            self._screen.blit(
                pygame.transform.scale(self._display, self._screen_size), self._TOP_LEFT)
            self._clock.tick(120)
            pygame.display.update()

    def add_task(self, task: Task) -> None:
        if not isinstance(task, Task):
            raise ValueError("Argument must be instance of Task")

        self._tasks.append(task)

    @property
    def display_size(self) -> tuple[float | int, float | int]:
        return self._display_size

    @property
    def screen_size(self) -> tuple[float | int, float | int]:
        return self._screen_size

    @property
    def window_caption(self) -> str:
        return self._window_caption

    @property
    def display_background(self) -> pygame.Surface | None:
        return self._background

    @display_background.setter
    def display_background(self, display_background: str | pygame.Surface | numpy.ndarray) -> None:
        self._background = Utils.load_surface(display_background, self._display_size)

    @property
    def board(self) -> Board | None:
        return self._board

    @property
    def collided_agents(self):
        return self._board.collided

    @property
    def agents(self) -> Iterable[Agent]:
        return self._board.agents
