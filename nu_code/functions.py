import sys
import numpy as np
import pickle
import os
import time
import psutil
import pickle
import pandas as pd
import re
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
from library_snapshot import system_tools as tools
from library_snapshot import sim_analysis as analysis
from library_snapshot import parser as parse

################## 
#### Parsing settings
################## 
def parse_parameters():
    parameters = parse.parse_parameter_file('./settings.ini')
    # simulation = parameters['simulation']
    bulk_species_ini = parameters['bulk_species']
    save_path = parameters['save_path']
    sim_path =parameters['file_path']
    ngrid_min =parameters['ngrid_min']
    ngrid_max = parameters['ngrid_max']
    ngrid_step = parameters['ngrid_step']
    mass_limit = parameters['mass_limit']
    
    return bulk_species_ini, save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, mass_limit



#### loading data
# def load_data_halo(sim_size, sim, snap_num, sim_path, halo_file, obj, verbose=True):
#     if verbose:
#         print("Loading data for bulk species '{}' in simulation '{}' at snapshot '{}'".format(sim, snap_num))

#     if verbose:
#         # print("\t Analyzing bulk species: {}".format(bulk_species))
#         print("\t Analyzing snapshot number: {}".format(snap_num))
#         print("\t Analyzing simulation: {}".format(sim))
#         # print("\t bulk chosen to : {}".format(bulk_species))
#     halo_address = halo_file
#     halo_all = obj.halo_selection(halo_address, 'all', 0)
#     pos = halo_all[:, :3]
#     vel = halo_all[:, 3:6]
#     mass = halo_all[:, 6]
#     if verbose:
#         print("\033[32m \t  number of  halos: " + str(np.shape(halo_all)[0]) + " \033[0m")

#     return pos, vel, mass # return the loaded particles or halo data

#### loading data
def load_data(bulk_species, sim_path, obj, mass_limit=1):
    # particles read
    if bulk_species == 'cdm':
        # snapshot = sim_path  + '/' + sim + '/output/snap00' + str(snap_num) + '_cdm'
        snapshot = sim_path
        ptype = [1]
        cdm_pcl = obj.gadget_load(snapshot, ptype)
        pos = cdm_pcl[0]
        vel = cdm_pcl[1]
    elif bulk_species == 'nu':
        ptype = [1]
        # snapshot = sim_path + '/' + sim + '/output/snap00' + str(snap_num) + '_ncdm0'
        snapshot = sim_path
        nu_pcl = obj.gadget_load(snapshot, ptype)
        pos = nu_pcl[0]
        vel = nu_pcl[1]
    elif bulk_species == 'halo':
        halo_address = sim_path
        halo_all = obj.halo_selection(halo_address, '+', mass_limit)
        pos = halo_all[:, :3]
        vel = halo_all[:, 3:6]
        # mass = halo_all[:, 6]

    return pos, vel # return the loaded particles or halo data


# Define function to measure memory and time usage
def print_usage(start_time, start_mem, msg):
    end_time = time.time()
    end_mem = psutil.Process().memory_info().rss
    walltime = end_time - start_time
    mem_usage = end_mem - start_mem
    print(f"\033[1m\033[94mWalltime:\033[0m \033[94m{walltime:.2f} seconds,\033[0m \033[1m\033[94mMemory usage:\033[0m \033[94m{mem_usage / (1024 ** 2):.2f} MB\033[0m"+msg)

def print_warning(text):
    print(f"\033[1m\033[94m WARNING:\033[0m {text}")
    
# def print_info(text):
#     print(f"\033[34m\033[97m {text} \033[0m")
    
def print_error(text):
    print(f"\033[1m\033[94m ERROR: \033[0m {text}")
    sys.exit(1)
    

def create_empty_dataframe(column_names):
    return pd.DataFrame(columns=column_names)

def generate_new_string(file_address):
    # Extract the relevant parts from the file address using regular expressions
    match_with_halos = re.search(r"/L_(\w+)_Ngrid_(\w+)/(\d+\.\d+)ev/halos/(.+)/", file_address)
    match_without_halos = re.search(r"/L_(\w+)_Ngrid_(\w+)/(\d+\.\d+)ev/output/(.+)", file_address)

    if match_with_halos:
        L_value = match_with_halos.group(1)
        Ngrid_value = match_with_halos.group(2)
        energy_value = match_with_halos.group(3)
        snap_cdm_string = match_with_halos.group(4)

    elif match_without_halos:
        L_value = match_without_halos.group(1)
        Ngrid_value = match_without_halos.group(2)
        energy_value = match_without_halos.group(3)
        snap_cdm_string = match_without_halos.group(4)

    else:
        return "Pattern not found in the file address."

    # Remove all occurrences of '/'
    snap_cdm_string = snap_cdm_string.replace('/', '_')

    # Remove the '.out' if present at the end of the snap_cdm_string
    snap_cdm_string = snap_cdm_string.replace('.out', '')
    snap_cdm_string = snap_cdm_string.replace('.list', '')

    # Create the new string in the desired format
    new_string = f"{energy_value}ev_L_{L_value}_Ngrid_{Ngrid_value}_{snap_cdm_string}"
    return [new_string, L_value, energy_value]


    
def save_dataframe(df, simulation, bulk_species, metadata, ngrid, save_path):
    # Add metadata
    # df.attrs['metadata'] = metadata
    data_to_save = {
        'metadata': metadata,
        'sub_box_data': df
    }

    # Save the dataframe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'/data_ngrid_'+str(ngrid)+'_sim_'+simulation+'.pickle', 'wb') as handle:
        pickle.dump(data_to_save, handle)

def read_file(file_path, msg=False):
    """
    Load data from a pickle file.

    Parameters:
        file_path (str): path to the pickle file.
        help (bool, optional): If True, print usage instructions.

    Returns:
        pandas.DataFrame: The loaded data.

    """
    if msg:
        print("To have the info: df[df['sub_box_index'].apply(lambda x: x == [1, 2, 1])].iloc[0]['halo_vel']")
        print("Also: df.attrs")
    
    with open(file_path, 'rb') as handle:
        df = pickle.load(handle)
        
    return df
