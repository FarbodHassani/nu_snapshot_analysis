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
from library_snapshot import readsnap
from collections import defaultdict
from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
##############
#### MPI part: #
##############

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
bulk_species_ini, save_path, file_path, ngrid_min, ngrid_max, ngrid_step, mass_limit = parse_parameters()
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
info_files=  generate_new_string(file_path[0])
simulation =info_files[0]
boxsize = np.double(info_files[1])
sim_type = info_files[2]+"ev"
if sim_type=="0.0ev":
    if bulk_species_ini[0] == "nu":
        if rank == 0:
            print_error("In the case of LCDM we don't have nu snapshots!")

obj = analysis.sim(boxsize);

     

###################################
# main part:
###################################
if bulk_species_ini[0] != 'halo': # In the case of neutrinos and cdm we need to know how many sub-files we have so that to loop over
    head = readsnap.snapshot_header(file_path[0])
    num_files = head.filenum # 
    ptype =  head.format; # type of gadget 2 format
    if rank == 0:
        print("number of gadget files to be loaded: "+str(num_files))

# Loop over the assigned ngrid_list for this process
ngrid_list_split = np.array_split(ngrid_list, size)
for ngrid in ngrid_list_split[rank]:
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss
    metadata = {'simulation': simulation, 'boxsize': boxsize, 'ngrid': ngrid, 'dx':  boxsize/ngrid, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step  
                , 'file_path': file_path, 'bulk_species': bulk_species_ini, 'ngrid': ngrid, 'save_path': save_path, 'sum_b_M':'<(bias_h + (mass)_h/(1.3e14))^0.85> average in each sub-box','sum_b_M_vel_h':'<(bias_h + (mass)_h/(1.3e14))^0.85 * v_h> average in each sub-box'}
    
    sub_box_data = {}  # Dictionary to store sub-box data
    coeff = ngrid / boxsize;
    #####################################
    ## In the case of halo catalogues!
    #####################################
    if bulk_species_ini[0] == 'halo':
        if rank == 0:
            print("The halo catalogue is loading in "+file_path[0]," mass_cut= 10^",np.round(np.log10(mass_limit),1))
        pos, vel, masses = load_data_halo(file_path[0], obj, mass_limit);
        for pcl in range(np.shape(pos)[0]):
            x_index = int(np.floor(pos[pcl, 0] * coeff))
            y_index = int(np.floor(pos[pcl, 1] * coeff))
            z_index = int(np.floor(pos[pcl, 2] * coeff))
            sub_box_index = (x_index, y_index, z_index)
            ###  mass of the halo
            mass = masses[pcl];
            bias_h = bias.haloBias(mass, model = 'sheth01', z = 0.0, mdef = '200m')
            # Check if sub-box exists in the dictionary, if not, initialize it
            if sub_box_index not in sub_box_data:
                sub_box_data[sub_box_index] = {
                    'sum_bulk_vel': 0.0,
                    'sum_b_M': 0.0, # (bias_h + (mass)_h/(1.3e14))^0.85 average in each sub-box
                    'sum_b_M_vel_h': 0.0, #(bias_h + (mass)_h/(1.3e14))^0.85* vh/2. average in each sub-box
                    'N': 0
                }
    
            # Accumulate velocity and number of particles in the sub-box
            sub_box_data[sub_box_index]['sum_bulk_vel'] += vel[pcl]
            sub_box_data[sub_box_index]['sum_b_M'] += (bias_h + (mass/(1.3*1.e14))**(0.85));
            sub_box_data[sub_box_index]['sum_b_M_vel_h'] += (bias_h + (mass/(1.3*1.e14))**(0.85))*vel[pcl];
            sub_box_data[sub_box_index]['N'] += 1
    #####################################
    ## In the case of cdm/nu !
    #####################################  
    else:
        ## We loop over gadget files and load them one-by-one which is more efficent memory-wise
        for num in range(num_files):
            pos, vel = load_data_gadget(file_path[0]+"."+str(num), ptype, obj);  
            if rank == 0:
                head = readsnap.snapshot_header(file_path[0]+"."+str(num))
                print("The file "+file_path[0]+"."+str(num),"is loading, file number", str(num),", type:"+bulk_species_ini[0],", number of pcl to be laoded: ",str(head.npart),", loaded num of particles:"+str(np.shape(pos)[0]))
        
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

    ### After looping over all particles we save the data!
    save_dataframe(sub_box_data, simulation, bulk_species_ini[0] , metadata, ngrid, save_path)
    print_usage(start_time, start_mem, f', n_grid={ngrid} Finished!')

# Synchronize all processes before finishing
comm.Barrier()

# Print total time and memory for all processes
if rank == 0:
    print_usage(start_time_all, start_mem_all, '- Total time and memory!')

