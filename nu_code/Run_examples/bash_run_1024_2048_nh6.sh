#!/bin/bash

# Define the number of MPI processes
NUM_PROCESSES=25

# Run the script with MPI
nohup mpiexec -n $NUM_PROCESSES python run_sims_1024_2048_nh6.py > out_run_sims_1024_2048_nh6.out &
