from setuptools import setup, find_packages


setup(
    name="occupancy",
    description="Count the number of people with ultrasonic sensor",
    author="Peter Todorov",
    author_email="peter.todorov@live.com",
    license="MIT",
    install_requires=open("requirements.txt").read().splitlines(),
    packages=find_packages(),
)
