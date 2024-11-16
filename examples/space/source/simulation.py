import numpy
import eevolve

from examples.space.source.space_agent import SpaceAgent


class Simulation:
    WINDOW_SIZE = (1280, 720)
    WINDOW_CAPTION = "Gravity"
    WINDOW_BACKGROUND = numpy.full((*WINDOW_SIZE, 3), (123, 123, 123), dtype=numpy.uint8)

    def __init__(self, agents_number: int) -> None:
        self._game = eevolve.Game(self.WINDOW_SIZE, self.WINDOW_SIZE,
                                  self.WINDOW_CAPTION, self.WINDOW_BACKGROUND, 15)
        self._agents_number = agents_number

    @staticmethod
    def movement_handler(agent: SpaceAgent, dt: float) -> None:
        agent.accelerate_by(agent.velocity * dt)

    @staticmethod
    def init_mass_handler(agent: SpaceAgent) -> None:
        agent.mass = numpy.pi * agent.rect.width ** 2 / 4

    @staticmethod
    def board_handler(agent: eevolve.Agent) -> None:
        collisions = [(1, -1), (-1, 1), (1, -1), (-1, 1)]

        for direction in agent.collision_directions:
            x, y = collisions[direction]

            agent.velocity[0] *= x
            agent.velocity[1] *= y

    @staticmethod
    def collision_handler(pair: tuple[SpaceAgent, SpaceAgent]) -> None:
        agent_1, agent_2 = pair

        c_1 = numpy.array(agent_1.rect.center)
        c_2 = numpy.array(agent_2.rect.center)

        v_1 = numpy.array(agent_1.velocity)
        v_2 = numpy.array(agent_2.velocity)

        m_1 = agent_1.mass
        m_2 = agent_2.mass

        delta_v = v_1 - v_2
        delta_c = c_1 - c_2
        distance_squared = numpy.dot(delta_c, delta_c)

        agent_1.velocity = v_1 - (2 * m_2 / (m_1 + m_2)) * (numpy.dot(delta_v, delta_c) / distance_squared) * delta_c
        agent_2.velocity = v_2 - (2 * m_1 / (m_1 + m_2)) * (numpy.dot(-delta_v, -delta_c) / distance_squared) * (
            -delta_c)

    def run(self) -> None:
        # TODO: Movement vectors for Agents, and internal task for movement, this will reduce need in BoardTasks
        # TODO: Also it is good approach to create some variables and then add internal Tasks to perform actions

        tasks = (
            eevolve.AgentMovementTask(self.movement_handler, 0, execution_number=1,
                                      priority=eevolve.HIGHEST_TASK_PRIORITY + 2),
            eevolve.BorderCollisionTask(self.board_handler, 0, priority=eevolve.HIGHEST_TASK_PRIORITY + 2),
            eevolve.CollisionTask(self.collision_handler, 0, priority=eevolve.HIGHEST_TASK_PRIORITY + 2),
            eevolve.AgentTask(self.init_mass_handler, 0, execution_number=1),
        )

        agent = SpaceAgent(0.0, agent_surface="examples/space/assets/star.png")
        agent_generators = {
            "size": eevolve.NumbersGenerator.hypercube_generator(self._agents_number, 2, 10.0, 10.0, dtype=int),
            "velocity": eevolve.NumbersGenerator.normal_generator(self._agents_number, shape=(2,), offset=0.0,
                                                                  scaler=15.0),
        }

        self._game.add_tasks(tasks)
        self._game.add_agents(self._agents_number,
                              eevolve.AgentGenerator.like_with_generators(agent, self._agents_number, agent_generators))
        self._game.run()
