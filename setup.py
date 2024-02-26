from setuptools import setup, find_packages

setup(
    name="CovNet",
    version="1.0",
    author="Joe Adamo",
    author_email="jadamo@arizona.edu",
    description="Neural network emulator to generate covariance matrices for cosmological data analysis",
    packages=find_packages(),
    python_requires='>=3.5,<3.9',
    install_requires=["build",
                      "numpy",
                      "scipy",
                      "torch",
                      "PyYAML",
                      "easydict",
                      "Cython",
                      "classy",
                      "mpi4py",
                      "six",
                      #"nbodykit", #<- not an official requirnment due to not workng on mac M1
    ],
)