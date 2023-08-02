#!/bin/bash

# this is a simple script that analyzes montepython chains, so I don't have to type out 
# directory names and settings stuff every time

base_dir="/Users/JoeyA/Research/"
#output_folder="/home/joeadamo/Research/CovNet/chains/MontePython/test-2/"
output_folder=$base_dir"CovNet/chains/MontePython/simulated-data/fixed-100/"
#output_folder="/home/joeadamo/Research/CovNet/chains/MontePython/montepython-data/multinest/"

./montepython/MontePython.py info $output_folder