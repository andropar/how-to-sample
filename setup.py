from setuptools import find_packages, setup

setup(
    name="utils",
    version="0.1",
    description="Module for creating controversial stimuli",
    author="Johannes Roth",
    author_email="johannes@roth24.de",
    packages=["how_to_sample"],
    package_dir={"": "src"},
)
