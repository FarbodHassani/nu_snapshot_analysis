#!/bin/bash

module load python
script_name="./main.py"

# command to run the python script
#python $script_name | tee info.txt                                          
mpirun -n 4 python $script_name | tee info.txt 