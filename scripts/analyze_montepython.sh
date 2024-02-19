#!/bin/bash

# this is a simple script that analyzes montepython chains, so I don't have to type out 
# directory names and settings stuff every time

#base_dir="/Users/JoeyA/Research/"
base_dir="/home/joeadamo/Research/"

output_folder=$base_dir"CovNet/chains/MontePython/beutler-data/vary-nd/"

./../montepython/MontePython.py info $output_folder