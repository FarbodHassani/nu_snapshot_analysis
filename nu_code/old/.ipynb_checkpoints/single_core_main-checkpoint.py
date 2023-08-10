from functions import parse_parameters
from functions import load_data
from functions import print_usage
from functions import create_empty_dataframe
from functions import save_dataframe
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
start_time = time.time()
start_mem = psutil.Process().memory_info().rss
sim_size_ini, boxsize, bulk_species_ini, snap_num_ini, sim_type_ini, \
save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, halo_file = parse_parameters()
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
obj = analysis.sim(boxsize);
pos, vel, mass = load_data(sim_size_ini[0], sim_type_ini[0], bulk_species_ini[0], snap_num_ini[0], sim_path, halo_file, obj, verbose=True);
print_usage(start_time, start_mem, ', Initialization time!')

############
# main part:
############
start_time = time.time()
start_mem = psutil.Process().memory_info().rss

for ngrid in ngrid_list: 
    sub_box_lists = obj.All_sub_box(ngrid);
    df = create_empty_dataframe(['halo_vel', 'halo_mass', 'bulk_vel' ,'halo_pos', 'sub_box_index']);
    
    for i in range(np.shape(sub_box_lists)[0]):
        sub_box_index = sub_box_lists[i]
        halos_i = obj.halos_in_cell(ngrid, pos, vel, mass, sub_box_index) # Gives the information of halos in a cell defined by sub-box index!
        bulk_vel_halos_i = obj.velocity_bulk(ngrid, pos, vel, sub_box_index)[:3] # Bulk velocities using halos
        data_saved = {
            'halo_vel':halos_i[:,3:6],
            'halo_mass':halos_i[:,6],
            'bulk_vel':bulk_vel_halos_i[0],
            'halo_pos': halos_i[:,0:3],
            'sub_box_index': sub_box_index
              }
        df = df.append(data_saved, ignore_index=True)
        # print_usage(start_time, start_mem, 'sub_box='+str(i))

    save_dataframe(df, sim_size_ini, boxsize, ngrid, bulk_species_ini, snap_num_ini, sim_type_ini, save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, halo_file)
    print_usage(start_time, start_mem, ', n_grid= '+str(ngrid)+' Finished!')