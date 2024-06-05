from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

print(required)

setup(
    name="pyBregMan",
    version="0.0.1",
    packages=find_packages(),
    requires=required,
)
