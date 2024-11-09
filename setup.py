from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="eevolve",
    version="0.0.10",
    author="Isak Volodymyr",
    author_email="volodymyr.o.isak@gmail.com",
    description="Evolution Algorithms Playground",
    long_description=long_description,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy==2.1.2",
        "pygame-ce==2.5.2",
    ]
)
