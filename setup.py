from setuptools import setup, find_packages

setup(
    name="CovNet",
    version="1.0",
    author="Joe Adamo",
    author_email="jadamo@arizona.edu",
    description="Neural network emulator to generate covariance matrices for cosmological data analysis",
    packages=find_packages(),
    install_requires=["build",
                      "numpy",
                      "scipy",
                      "torch >= 1.12.1",
                      "PyYAML",
                      "easydict",
                      "camb >= 1.3.5",
                      "classy"
    ],
)