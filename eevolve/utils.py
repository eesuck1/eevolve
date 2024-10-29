import os

import numpy
import pygame


class Utils:
    @staticmethod
    def load_surface(surface: str | pygame.Surface | numpy.ndarray,
                     desired_size: tuple[int | float, int | float]) -> pygame.Surface | None:
        result = None
        if (isinstance(surface, str)
                and os.path.exists(surface)
                and pygame.image.get_extended()):
            try:
                image = pygame.image.load(surface)
                result = pygame.transform.scale(image, desired_size).convert()
            except pygame.error:
                print("[ERROR] Surface image could not be loaded.")
        elif isinstance(surface, pygame.Surface):
            result = surface.convert()
        elif isinstance(surface, numpy.ndarray):
            try:
                result = pygame.surfarray.make_surface(surface).convert()
            except pygame.error:
                print("[ERROR] Surface image could not be loaded.")
        else:
            print("[ERROR] Surface image could not be loaded. Unsupported format")

        return result if result is not None else pygame.Surface(desired_size).convert()

    @staticmethod
    def clip(value: int | float, a: int | float, b: int | float) -> int | float:
        value = a if value < a else value
        value = b if value > b else value

        return value
