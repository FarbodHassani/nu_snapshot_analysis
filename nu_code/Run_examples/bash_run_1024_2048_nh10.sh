#!/bin/bash

# Define the number of MPI processes
NUM_PROCESSES=1

# Run the script with MPI
nohup mpiexec -n $NUM_PROCESSES python run_sims_1024_2048_nh10.py > out_run_sims_1024_2048_nh10.out &
