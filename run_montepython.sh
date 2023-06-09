#!/bin/bash

# this is a simple script that runs montepython for ease of access

export OMP_NUM_THREADS=4

base_dir="/home/joeadamo/Research/"
conf_file=$base_dir"Software/Montepython/default.conf"
param_file=$base_dir"lss_montepython/input/boss_ngc_z3.param"
output_folder=$base_dir"CovNet/chains/MontePython/simulated-data/gaussian-emulator-AE/"

#covmat="/home/joeadamo/Research/CovNet/chains/MontePython/beutler-data/old-likelihood/initial-marg/initial-beutler.covmat"
covmat="/home/joeadamo/Research/CovNet/chains/MontePython/simulated-data/vary-gaussian/test.covmat"
#covmat="/home/joeadamo/Research/CovNet/chains/MontePython/simulated-data/vary-no-determinant-1/vary-no-determinant.covmat"

log_file=$output_folder"log.param"

# can be "global", "sequential" or "fast"
# "global" - varies all parameters at the same time (normal for MH)
# "sequential" - varies one parameter at a time (good for debugging / initial runs?)
# "fast" - varies cosmo and nuisance parameters seperately (DO NOT USE!)
jump_mode="global"
N=15000

# flag to restart a run using -r
# need to specify the name of the lowest-number chain (so example date_N__1.txt)
# restart=$output_folder"2023-04-04_15000__1.txt"

# flag to restart from a best-fit location of a previous run
# best_fit="/home/joeadamo/Research/CovNet/chains/MontePython/test-2/test-2.bestfit"

# flag to run with multinest instead
multinest="NO"

#rm $log_file
ls $output_folder

if [ $multinest = "YES" ]; then
    mpirun -np 4 ./montepython/MontePython.py run --conf $conf_file -p $param_file -o $output_folder -j $jump_mode -m NS
# run montepython based on the above parameters being set
else if [ -z ${restart+x} ]; then
    # starting a new run
    if [ -z ${covmat+x} ]; then
        # no input covariance
        echo "staring without prior parameter covariance"
        mpirun -np 4 ./montepython/MontePython.py run --conf $conf_file -p $param_file -o $output_folder -j $jump_mode -N $N
    else
        # with input covariance
        echo "starting with prior parameter covariance"
        mpirun -np 4 ./montepython/MontePython.py run --conf $conf_file -p $param_file -o $output_folder -j $jump_mode -c $covmat -N $N
        #mpirun -np 4 ./montepython/MontePython.py run --conf $conf_file -p $param_file -o $output_folder -j $jump_mode -c $covmat -N $N -b $best_fit
    fi
else
    # restarting a previous run
    mpirun -np 4 ./montepython/MontePython.py run --conf $conf_file -p $param_file -o $output_folder -j $jump_mode -c $covmat -N $N -r $restart
fi
fi