from setuptools import setup, find_packages

setup(
    name="CovNet",
    version="1.0",
    author="Joe Adamo",
    author_email="jadamo@arizona.edu",
    packages=find_packages(),
    install_requires=[
                      "numpy",
                      "scipy",
                      "torch >= 1.12.1",
                      "PyYAML",
                      "camb >= 1.3.5"
    ],
)