import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="hummingbird-classifier",
    version="0.1.0",
    author="Michele Volpi, Luca Pegoraro",
    author_email="mivolpi@ethz.ch, luca.pegoraro@wsl.ch",
    description=(
        "A simple DL-based binary classifier for presence/absence of hummingbirds in complex visual scenes"
    ),
    license="BSD",
    keywords="hummingbirds detection classification deep-learning computer-vision",
    url="https://gitlab.renkulab.io/biodetect/hummingbird-classifier",
    packages=[],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 -- Alpha",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: BSD License",
    ],
)
