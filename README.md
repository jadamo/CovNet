# CovNet
This repository contains the code required to design, train, and test neural networks to generate covariance matrices at different cosmologies.

## Getting Started 
1. Install CLASS-PT (https://github.com/Michalychforever/CLASS-PT) and follow the instillation instructions for the `classy` python extension there.
2. If you would like to enable GPU functionality for network training, install the corresponding `PyTorch` version here: https://pytorch.org/get-started/locally/. If your machine doesn't have a GPU, you can skip this step.
3. Download this repository to your location of choice.
4. In the base directory, run `python -m pip install .`, which should install this repository as a package you can call anywhere on your machine.

## Using the Code
A description of how to use this repository is given in the [Wiki](https://github.com/jadamo/CovNet/wiki/Covariance-Matrix-Emulator-Workflow). 

## Citing the Code
Users of this code must cite the authors (TODO: update with paper link), CovaPT (https://arxiv.org/abs/1910.02914), and CLASS-PT (https://arxiv.org/abs/2004.10607)
