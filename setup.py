from setuptools import setup, find_packages


setup(
    author="Peter Todorov",
    author_email="peter.todorov@live.com",
    license="No Redistribution Clause",
    install_requires=open("requirements.txt").read().splitlines(),
    packages=find_packages(),
)
