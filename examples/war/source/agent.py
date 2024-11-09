import numpy
import pygame

import eevolve


class WarAgent(eevolve.Agent):
    def __init__(self, agent_size: tuple[int | float, int | float],
                 agent_position: tuple[int | float, int | float] | numpy.ndarray, agent_name: str,
                 agent_surface: str | pygame.Surface | numpy.ndarray, brain: eevolve.Brain):
        super().__init__(agent_size, agent_position, agent_name, agent_surface, brain)

    @property
    def color(self) -> str:
        return self.name.split("_")[0]
