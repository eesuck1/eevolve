from examples.space.source.simulation import Simulation


def run() -> None:
    simulation = Simulation(150)
    simulation.run()


if __name__ == '__main__':
    run()
