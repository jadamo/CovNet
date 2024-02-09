#!/bin/bash

# this script runs a batch of mcmc chains sequentially

export OMP_NUM_THREADS=3

#base_dir="/Users/JoeyA/Research/"
base_dir="/home/joeadamo/Research/"
conf_file=$base_dir"Software/Montepython/default.conf"
param_file=$base_dir"Software/Montepython/input/boss_ngc_z3.param"
cfg_file=$base_dir"Software/Montepython/montepython/likelihoods/ngc_z3_noisy/ngc_z3_noisy.data"

covmat=$base_dir"CovNet/chains/MontePython/noisy-data/vary-noisy-1/vary-noisy-1.covmat"

# can be "global", "sequential" or "fast"
# "global" - varies all parameters at the same time (normal for MH)
# "sequential" - varies one parameter at a time (good for debugging / initial runs?)
# "fast" - varies cosmo and nuisance parameters seperately (DO NOT USE!)
jump_mode="global"
N=2500

# flag to restart from a best-fit location of a previous run
# best_fit="/home/joeadamo/Research/CovNet/chains/MontePython/test-2/test-2.bestfit"


#rm $log_file
ls $output_folder

for i in {1..25}
do 
    output_folder=$base_dir"CovNet/chains/MontePython/noisy-data/vary-short-$i/"
    log_file=$output_folder"log.param"
    echo $output_folder

    # set up directory
    if [ -d $output_folder ]; then
        echo "Directory already exists. Removing what's there..."
        rm -rf $output_folder
        # if [ -e $output_folder"log.param" ]; then
        #     echo "Previous run saved to this folder! Removing files..."
        #     rm -rf $output_folder"*"
        # fi
    fi

    mkdir $output_folder

    # edit index variable in ngc_z3_noisy.param file to change data vector to use
    sed -i 's/ngc_z3_noisy.idx=.*/ngc_z3_noisy.idx='$i'/' $cfg_file

    # run the chain
    mpirun -np 4 ./../montepython/MontePython.py run --conf $conf_file -p $param_file -o $output_folder -j $jump_mode -c $covmat -N $N
done