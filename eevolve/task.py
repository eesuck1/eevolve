from typing import Callable, Any

from eevolve.agent import Agent
from eevolve.board import Board


class Task:
    """
    A class representing a scheduled task that can be called at a specified interval in milliseconds,
    if zero function will be called at the end of each frame.

    This class allows you to define a function that will be executed periodically, with the ability to pass arguments
    to that function while creating or create specific Task subclass with predefined signature.

    Example:

    def game_task_handler(game: Game) -> None:
        print(game.agents)

    def board_task_handler(board: Board) -> None:
        print(board.agents)

    def frame_end_task_handler(game: Game) -> None:
        if len(game.agents) == 0:
            game.add_agents(10, AgentGenerator.default(game, 10))

    def agent_task_handler(agent: Agent) -> None:
        agent.move_by((5, 5))

    def collision_task_handler(collision_pair: tuple[Agent, Agent]) -> None:
        agent_1, agent_2 = collision_pair

        print(f"{agent_1} collide {agent_2}")

    game_task = Task(game_task, 1000, game)

    board_task = BoardTask(board_task_handler, 100)

    frame_end_task = FrameEndTask(frame_end_task_handler, 2500, game)

    agent_task = AgentTask(agent_task_handler, 500)

    collision_task = CollisionTask(collision_task_handler, 0)

    tasks = (game_task, board_task, frame_end_task, agent_task, collision_task)

    game.add_tasks(tasks)

    game.run()

    :param function:
        The function to be executed by the task.

    :param period_ms:
        The period (in milliseconds) at which the task should be executed.

    :param execution_number:
        The number of times the task will be performed

    :param args:
        Positional arguments to pass to the function when called.

    :param kwargs:
        Keyword arguments to pass to the function when called.
    """

    def __init__(self, function: Callable[..., Any], period_ms: int, execution_number: int = -1,
                 *args, **kwargs) -> None:
        self._function = function
        self._period_ms = period_ms
        self._execution_number = execution_number
        self._timer = 0

        self._args = args
        self._kwargs = kwargs

    @property
    def period(self) -> float:
        return self._period_ms

    @property
    def timer(self) -> float:
        return self._timer

    @timer.setter
    def timer(self, value: float) -> None:
        self._timer = value

    @property
    def is_dead(self) -> bool:
        return self._execution_number == 0

    def __call__(self, *args, **kwargs) -> Any:
        if self._execution_number == 0:
            return

        result = self._function(*self._args, *args, **self._kwargs, **kwargs)

        if self._execution_number > 0:
            self._execution_number -= 1

        return result


class CollisionTask(Task):
    def __init__(self, function: Callable[[tuple[Agent, Agent]], None], period_ms: int, execution_number: int = -1, *args, **kwargs) -> None:
        super().__init__(function, period_ms, execution_number, *args, **kwargs)


class AgentTask(Task):
    def __init__(self, function: Callable[[Agent], None], period_ms: int, execution_number: int = -1, *args, **kwargs):
        super().__init__(function, period_ms, execution_number, *args, **kwargs)


class FrameEndTask(Task):
    def __init__(self, function: Callable, execution_number: int = -1, *args, **kwargs) -> None:
        super().__init__(function, 0, execution_number, *args, **kwargs)


class BoardTask(Task):
    def __init__(self, function: Callable[[Board], None], period_ms: int, execution_number: int = -1, *args, **kwargs):
        super().__init__(function, period_ms, execution_number, *args, **kwargs)


class PairTask(Task):
    def __init__(self, function: Callable[[tuple[Agent, Agent]], None], period_ms: int, execution_number: int = -1, *args, **kwargs):
        super().__init__(function, period_ms, execution_number, *args, **kwargs)
