from mpi4py import MPI
import numpy as np
import sys
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
from functions import parse_parameters
from functions import load_data_halo
from functions import load_data_pcl
from functions import print_usage
from functions import create_empty_dataframe
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
sim_size_ini, boxsize, bulk_species_ini, snap_num_ini, sim_type_ini, \
save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, halo_file = parse_parameters()
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
obj = analysis.sim(boxsize);
pos, vel, mass = load_data_halo(sim_size_ini[0], sim_type_ini[0], snap_num_ini[0], sim_path, halo_file, obj, verbose=False);
if sim_type_ini[0]!="0.0ev":
    pos_nu, vel_nu = load_data_pcl(sim_size_ini[0], sim_type_ini[0], 'nu', snap_num_ini[0], sim_path, halo_file, obj, verbose=False);
pos_cdm, vel_cdm = load_data_pcl(sim_size_ini[0], sim_type_ini[0], 'cdm', snap_num_ini[0], sim_path, halo_file, obj, verbose=False);
if rank==0:
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
    if sim_type_ini[0]!="0.0ev":
        df = create_empty_dataframe(['halo_vel', 'halo_mass', 'bulk_vel_halo', 'bulk_vel_cdm','bulk_vel_nu', 'halo_pos', 'sub_box_index'])
    else:
        df = create_empty_dataframe(['halo_vel', 'halo_mass', 'bulk_vel_halo', 'bulk_vel_cdm', 'halo_pos', 'sub_box_index'])

        
    halos_positions = obj.precalculate_halo_positions(ngrid, pos, vel, mass)
    bulk_vel_halos = obj.precalculate_bulk_velocities(ngrid, pos, vel)
    bulk_vel_cdm = obj.precalculate_bulk_velocities(ngrid, pos_cdm, vel_cdm)
    bulk_vel_nu = obj.precalculate_bulk_velocities(ngrid, pos_nu, vel_nu)

    for sub_box_index in sub_box_lists:
        halos_i = halos_positions[tuple(sub_box_index)]
        bulk_vel_halos_i = bulk_vel_halos[tuple(sub_box_index)]
        bulk_vel_cdm_i = bulk_vel_cdm[tuple(sub_box_index)]
        bulk_vel_nu_i = bulk_vel_nu[tuple(sub_box_index)]

        if sim_type_ini[0]!="0.0ev":
            bulk_vel_nu_i = obj.velocity_bulk(ngrid, pos_nu, vel_nu, sub_box_index)[:3]
            data_saved = {
                'halo_vel': halos_i[:,3:6],
                'halo_mass': halos_i[:,6],
                'bulk_vel_halo': bulk_vel_halos_i[0],
                'bulk_vel_cdm': bulk_vel_cdm_i[0],
                'bulk_vel_nu': bulk_vel_nu_i[0],
                'halo_pos': halos_i[:,0:3],
                'sub_box_index': sub_box_index
            }
        else:
            data_saved = {
                'halo_vel': halos_i[:,3:6],
                'halo_mass': halos_i[:,6],
                'bulk_vel_halo': bulk_vel_halos_i[0],
                'bulk_vel_cdm': bulk_vel_cdm_i[0],
                'halo_pos': halos_i[:,0:3],
                'sub_box_index': sub_box_index
            } 
        df = df.append(data_saved, ignore_index=True)

    # Barrier to ensure all cores have completed their calculations
    comm.Barrier()

    # Concatenate data from all cores
    all_data = comm.gather(df, root=0)

    if rank == 0:
        df = pd.concat(all_data, ignore_index=True)
        save_dataframe(df, sim_size_ini, boxsize, ngrid, bulk_species_ini, snap_num_ini, sim_type_ini, save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, halo_file)
        print_usage(start_time, start_mem, ', n_grid= ' + str(ngrid) + ' Finished!')

    # Clear dataframe before processing next ngrid
    df = None

if rank == 0:   
    print_usage(start_time_all, start_mem_all, '- Total time and memory!')
