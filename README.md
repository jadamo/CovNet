# CovNet
This package contains the code required to design, train, and test neural networks to generate covariance matrices at different cosmologies.

This package is compatable with **python 3.5 - 3.8**

## Installing the Code
This package should work for both Linux and MacOS. We recommend using anaconda for package management, but standalone pip should work as well. 

1. Install CLASS-PT (https://github.com/Michalychforever/CLASS-PT) and follow the instillation instructions for the `classy` python extension there.
2. Make sure you have a working MPI implimentation on your machine (ex. [openmpi](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html)). 
3. If you would like to enable GPU functionality for network training, make sure you are using python 3.8 and install the corresponding [PyTorch version](https://pytorch.org/get-started/locally/). If your machine doesn't have a GPU, you can skip this step.
4. Download this repository to your location of choice.
5. In the base directory, run `python -m pip install .`, which should install this repository as a package you can call anywhere on your machine.

NOTE: In order to calculate your own covariance window functions, you'll also need to install [nbodykit](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html#macos). This package is not configured to run on the new apple processors, thus it isn't in the package dependency list.

## Using the Code
A description of how to use CovNet is given [here](https://github.com/jadamo/CovNet/wiki/Covariance-Matrix-Emulator-Workflow). 

## Citing the Code
Users of this code must cite the authors (TODO: update with paper link), CovaPT (https://arxiv.org/abs/1910.02914), and CLASS-PT (https://arxiv.org/abs/2004.10607)
