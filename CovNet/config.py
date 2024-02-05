# -----------------------------------------------
# this file contains all necesary file paths to use this repository
import os

# Directory with window functions needed to calculate covariance matrices
# by default this is the "data" directory within CovNet itself
CovaPT_data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/data/"
# or you can hard-code the directory where your data resides
#CovaPT_data_dir = "/home/joeadamo/Research/CovNet/data/"
