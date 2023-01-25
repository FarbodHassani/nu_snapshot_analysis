#!/bin/bash

script_name="./python_run_alpha.py"

# command to run the python script
python $script_name | unbuffer -p tee info.txt
