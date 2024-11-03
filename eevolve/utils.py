import os

import numpy
import pygame


class Utils:
    @staticmethod
    def load_surface(surface: str | pygame.Surface | numpy.ndarray,
                     desired_size: tuple[int | float, int | float]) -> pygame.Surface | None:
        """
        Loads an image surface from a file, Pygame surface, or NumPy array
        and scales it to the desired size.

        If the input surface is a string, it checks if the file exists
        and if Pygame's extended image loading is available. If it is a
        Pygame surface, it simply converts it. If it is a NumPy array,
        it attempts to create a surface from the array.

        Example 1: Loading an image from a file


        surface = Utils.load_surface("path/to/image.png", (100, 100))

        Example 2: Using a Pygame Surface


        existing_surface = pygame.Surface((200, 200))

        scaled_surface = Utils.load_surface(existing_surface, (100, 100))

        Example 3: Using a NumPy array
        import numpy


        array_data = numpy.zeros((100, 100, 3), dtype=numpy.uint8)

        surface_from_array = Utils.load_surface(array_data, (100, 100))

        :param surface:
            The source of the image, which can be a file path (string),
            a Pygame surface, or a NumPy array representing the image data.

        :param desired_size:
            The desired size (width, height) to scale the image surface.

        :return:
            A Pygame surface scaled to the desired size if successful;
            otherwise, returns a new Pygame surface of the desired size
            filled with black.

        :raises ValueError:
            If the image cannot be loaded from the given surface or file path.
        """

        if (isinstance(surface, str)
                and os.path.exists(surface)
                and pygame.image.get_extended()):
            try:
                image = pygame.image.load(surface)
                result = pygame.transform.scale(image, desired_size).convert()
            except pygame.error:
                raise ValueError("Surface image could not be loaded.")
        elif isinstance(surface, pygame.Surface):
            result = surface.convert()
        elif isinstance(surface, numpy.ndarray):
            try:
                result = pygame.surfarray.make_surface(surface).convert()
            except pygame.error:
                raise ValueError("Surface image could not be loaded.")
        else:
            raise ValueError("Surface image could not be loaded. Unsupported format")

        return result if result is not None else pygame.Surface(desired_size).convert()

    @staticmethod
    def clip(value: int | float, a: int | float, b: int | float) -> int | float:
        """
        Clamps a given value between two bounds.

        Example 1:

        clamped_value = Utils.clip(10, 0, 5)

        :param value:
            The value to be clamped.

        :param a:
            The lower bound.

        :param b:
            The upper bound.

        :return:
            The clamped value, which will be equal to `a` if the original value
            was less than `a`, equal to `b` if it was greater than `b`,
            or the original value if it lies within the bounds.
        """

        value = a if value < a else value
        value = b if value > b else value

        return value
