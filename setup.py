from setuptools import setup, find_packages


setup(
    name="eevolve",
    version="0.0.10",
    author="Isak Volodymyr",
    author_email="volodymyr.o.isak@gmail.com",
    description="Evolution Algorithms Playground",
    packages=find_packages(),
    install_requires=[
        "numpy==2.1.2",
        "pygame-ce==2.5.2",
    ]
)
