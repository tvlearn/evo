from setuptools import setup, find_packages

setup(
    name="evo",
    packages=find_packages(exclude=("test", "examples")),
    zip_safe=False,
)
