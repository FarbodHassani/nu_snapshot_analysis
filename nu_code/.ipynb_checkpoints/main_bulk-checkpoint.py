from mpi4py import MPI
import numpy as np
import sys
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
from functions import parse_parameters
from functions import load_data
from functions import print_error
from functions import print_usage
from functions import create_empty_dataframe
from functions import generate_new_string
from functions import save_dataframe
from multiprocessing import Lock
import pickle
import numpy as np
import pandas as pd
import os
from library_snapshot import sim_analysis as analysis
import time
import psutil
## To run mpirun -n 4 main_bulk.py
############
# MPI
############
lock = Lock()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_cores = comm.Get_size()

############
# Parsing the settings
############
start_time_all = time.time()
start_mem_all = psutil.Process().memory_info().rss
bulk_species_ini, save_path, file_path, ngrid_min, ngrid_max, ngrid_step = parse_parameters()
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
info_files=  generate_new_string(file_path[0])
simulation =info_files[0]
boxsize = np.double(info_files[1])
sim_type = info_files[2]+"ev"
if sim_type=="0.0ev":
    if bulk_species_ini[0] == "nu":
        print_error("In the case of LCDM we don't have nu snapshots!")

obj = analysis.sim(boxsize);        
pos, vel = load_data(bulk_species_ini[0], file_path[0], obj);       
if rank==0:
    print("Loading data for bulk species '{}' in simulation '{}' with boxsize '{}'".format(bulk_species_ini[0], file_path[0], boxsize))
    print_usage(start_time_all, start_mem_all, ', Data initialized!')
    print("Number of cores being used:", n_cores)

# ###########
# main part:
# ###########

for ngrid in ngrid_list:
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss
    sub_box_lists = obj.All_sub_box(ngrid)
    n_subboxes = len(sub_box_lists)

    if n_cores > n_subboxes:
        n_cores_to_use = n_subboxes
    else:
        n_cores_to_use = n_cores

    subbox_per_core = n_subboxes // n_cores_to_use
    extra_subboxes = n_subboxes % n_cores_to_use

    if rank < extra_subboxes:
        start_index = (subbox_per_core + 1) * rank
        end_index = start_index + subbox_per_core + 1
    else:
        start_index = subbox_per_core * rank + extra_subboxes
        end_index = start_index + subbox_per_core

    sub_box_lists = sub_box_lists[start_index:end_index]
    df = create_empty_dataframe(['bulk_vel', 'sub_box_index'])
    metadata = {'simulation': simulation, 'boxsize': boxsize, 'ngrid': ngrid, 'dx':  boxsize/ngrid, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step  , 'file_path': file_path, 'bulk_species': bulk_species_ini, 'ngrid': ngrid, 'save_path': save_path}
    
    for sub_box_index in sub_box_lists:
        bulk_vel_i = obj.velocity_bulk(ngrid, pos, vel, sub_box_index)
        data_saved = {
            'bulk_vel': bulk_vel_i[0],
            'bulk_vel_error': bulk_vel_i[1],
            'Number of particles in the cell': bulk_vel_i[2],
            'sub_box_index': sub_box_index
        }
        df = df.append(data_saved, ignore_index=True)
    # Barrier to ensure all cores have completed their calculations
    comm.Barrier()
    # Concatenate data from all cores
    all_data = comm.gather(df, root=0)

    if rank == 0:
        df = pd.concat(all_data, ignore_index=True)
        save_dataframe(df, simulation, bulk_species_ini[0] , metadata, ngrid, save_path)
        print_usage(start_time, start_mem, ', n_grid= ' + str(ngrid) + ' Finished!')

    # Clear dataframe before processing next ngrid
    df = None

if rank == 0:   
    print_usage(start_time_all, start_mem_all, '- Total time and memory!')