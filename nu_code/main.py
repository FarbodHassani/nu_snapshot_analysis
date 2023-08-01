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


############
# Parsing the settings
############
start_time_all = time.time()
start_mem_all = psutil.Process().memory_info().rss
bulk_species_ini, save_path, file_path, ngrid_min, ngrid_max, ngrid_step, mass_limit = parse_parameters()
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
info_files=  generate_new_string(file_path[0])
simulation =info_files[0]
boxsize = np.double(info_files[1])
sim_type = info_files[2]+"ev"
if sim_type=="0.0ev":
    if bulk_species_ini[0] == "nu":
        print_error("In the case of LCDM we don't have nu snapshots!")

obj = analysis.sim(boxsize);
if bulk_species_ini[0] == 'halo':
    pos, vel = load_data(bulk_species_ini[0], file_path[0], obj, mass_limit);       
else:
    pos, vel = load_data(bulk_species_ini[0], file_path[0], obj);       

print("Loading data for bulk species '{}' in simulation '{}' with boxsize '{}'".format(bulk_species_ini[0], file_path[0], boxsize))
print("",np.shape(pos)[0], " number of particles to analyse")
print_usage(start_time_all, start_mem_all, ', Data initialized!')

# ###########
# main part:
# ###########
for ngrid in ngrid_list:
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss
    metadata = {'simulation': simulation, 'boxsize': boxsize, 'ngrid': ngrid, 'dx':  boxsize/ngrid, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step  , 'file_path': file_path, 'bulk_species': bulk_species_ini, 'ngrid': ngrid, 'save_path': save_path}
    
    sub_box_data = {}  # Dictionary to store sub-box data
    for pcl in range(np.shape(pos)[0]):
        x_index = int(np.floor(pos[pcl, 0] * ngrid / boxsize))
        y_index = int(np.floor(pos[pcl, 1] * ngrid / boxsize))
        z_index = int(np.floor(pos[pcl, 2] * ngrid / boxsize))
        sub_box_index = (x_index, y_index, z_index)

        # Check if sub-box exists in the dictionary, if not, initialize it
        if sub_box_index not in sub_box_data:
            sub_box_data[sub_box_index] = {
                'sum_bulk_vel': 0.0,
                'N': 0
            }

        # Accumulate velocity and number of particles in the sub-box
        sub_box_data[sub_box_index]['sum_bulk_vel'] += vel[pcl]
        sub_box_data[sub_box_index]['N'] += 1

    ### After looping over all particles we save the data!
    save_dataframe(sub_box_data, simulation, bulk_species_ini[0] , metadata, ngrid, save_path)
    print_usage(start_time, start_mem, f', n_grid={ngrid} Finished!')

print_usage(start_time_all, start_mem_all, '- Total time and memory!')