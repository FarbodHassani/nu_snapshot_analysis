#!/bin/bash

module load python
script_name="./main.py"

# command to run the python script
python $script_name > info.txt
