import sys
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/JD_figs_tests_data/')
from ReadHalos import *
from ReadParticles import *
from mpi4py import MPI
import numpy as np
import sys
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
from functions import *
from multiprocessing import Lock
import pickle
import numpy as np
import pandas as pd
import os
from library_snapshot import sim_analysis as analysis
import time
import psutil

# #############
# ### MPI part: #
# #############

# Get the rank and size of the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Divide the ngrid_list among the processes

############
# Parsing the settings
############
start_time_all = time.time()
start_mem_all = psutil.Process().memory_info().rss
bulk_species_ini, save_path, file_path, ngrid_min, ngrid_max, ngrid_step, mass_limit, boxsize, sim_type = parse_parameters_JD()
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
info_files=  generate_new_string(file_path[0])
# simulation =info_files[0]
sim_type = sim_type[0]
if sim_type=="0.0ev":
    if bulk_species_ini[0] == "nu":
        if rank == 0:
            print_error("In the case of LCDM we don't have nu snapshots!")

obj = analysis.sim(boxsize);
if rank == 0:
    print("Loading data for bulk species '{}' in simulation '{}' with boxsize '{}'".format(bulk_species_ini[0], file_path[0], boxsize))
    print_usage(start_time_all, start_mem_all, ', Data initialized!')
# ###########
# main part:
# ###########

# Loop over the assigned ngrid_list for this process
ngrid_list_split = np.array_split(ngrid_list, size);
simulation = sim_type+"_L_500_Ngrid_6144_"+bulk_species_ini[0]
rank_total = 512;
start_time = time.time()
start_mem = psutil.Process().memory_info().rss
for ngrid in ngrid_list_split[rank]:
    metadata = {'simulation': simulation, 'boxsize': boxsize, 'ngrid': ngrid, 'dx':  boxsize/ngrid, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step, 'file_path': file_path, 'bulk_species': bulk_species_ini, 'ngrid': ngrid, 'save_path': save_path}

    sub_box_data = {}  # Dictionary to store sub-box data
    coeff = ngrid / (boxsize+0.01);
    ### Loop over all the simulation files
    for rank_id in range(rank_total):
        if bulk_species_ini[0] == 'cdm':
            input_file = file_path[0]+"/0.000xv"+str(rank_id)+".dat"
            file_data = ReadParticleFile(input_file)
            pos = np.vstack((file_data[0], file_data[1], file_data[2])).T
            vel = np.vstack((file_data[3], file_data[4], file_data[5])).T
        if bulk_species_ini[0] == 'nu':
            input_file = file_path[0]+"/0.000xv"+str(rank_id)+"_nu.dat"
            file_data = ReadParticleFile(input_file)
            pos = np.vstack((file_data[0], file_data[1], file_data[2])).T
            vel = np.vstack((file_data[3], file_data[4], file_data[5])).T
        elif bulk_species_ini[0] == 'halo':
            input_file = file_path[0]+"/0.000halo"+str(rank_id)+".dat"
            a = 1.;
            file_data = ReadHaloFile_data(input_file, a)
            cond = (file_data[6] >= mass_limit)  # Assuming file_data[6] contains the values you want to compare
            pos = np.vstack((file_data[0][cond], file_data[1][cond], file_data[2][cond])).T
            vel = np.vstack((file_data[3][cond], file_data[4][cond], file_data[5][cond])).T
        ### Need to prepare the positions and velocities before this loop
        for pcl in range(np.shape(pos)[0]):
            x_index = int(np.floor(pos[pcl, 0] * coeff))
            y_index = int(np.floor(pos[pcl, 1] * coeff))
            z_index = int(np.floor(pos[pcl, 2] * coeff))
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

            if rank == 0:
                if(rank_id%10==0):
                    print_usage(start_time, start_mem, "; file number "+str(rank_id)+" is loaded for N_grid: "+str(ngrid))
                    print("%%%%%%%%%%%%")

    ### After looping over all particles we save the data!
    save_dataframe(sub_box_data, simulation, bulk_species_ini[0] , metadata, ngrid, save_path)
    print_usage(start_time, start_mem, f', n_grid={ngrid} Finished!')

# Synchronize all processes before finishing
comm.Barrier()

# Print total time and memory for all processes
if rank == 0:
    print_usage(start_time_all, start_mem_all, '- Total time and memory!')
