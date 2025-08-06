#!/bin/bash
eval "$(/home/hep/us322/miniforge3/bin/conda shell.bash hook)"
conda activate madminer_env

seed=25

python 05_train_network.py -p ${3} -rid dense_s${seed}_${3}_f${2} -f ${2} -c${1} -s ${seed}
