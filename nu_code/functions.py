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
from library_snapshot import readsnap
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')
from ReadHalos import *
from ReadParticles import *



################## 
#### Parsing settings
################## 
def parse_parameters(file='./settings.ini'):
    parameters = parse.parse_parameter_file(file)
    # simulation = parameters['simulation']
    bulk_species_ini = parameters['bulk_species']
    save_path = parameters['save_path']
    sim_path =parameters['file_path']
    ngrid_min =parameters['ngrid_min']
    ngrid_max = parameters['ngrid_max']
    ngrid_step = parameters['ngrid_step']
    mass_limit = parameters['mass_limit']
    
    return bulk_species_ini, save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, mass_limit

def parse_parameters_JD(file='./settings.ini'):
    parameters = parse.parse_parameter_file(file)
    # simulation = parameters['simulation']
    bulk_species_ini = parameters['bulk_species']
    save_path = parameters['save_path']
    sim_path =parameters['file_path']
    ngrid_min =parameters['ngrid_min']
    ngrid_max = parameters['ngrid_max']
    ngrid_step = parameters['ngrid_step']
    mass_limit = parameters['mass_limit']
    boxsize = parameters['boxsize']
    sim_type = parameters['sim_type']

    return bulk_species_ini, save_path, sim_path, ngrid_min, ngrid_max, ngrid_step, mass_limit, boxsize, sim_type


def load_data_gadget(sim_path, ptype ,obj):
    # particles read
    pos = readsnap.read_block(sim_path, "POS ", ptype)/1e3 #positions in Mpc/h #readsnap.read_block(snapshot, "POS ", ptype, True, 0, False, False,[0,num,0,0,0,0])/1e3
    vel = readsnap.read_block(sim_path, "VEL ", ptype)     #peculiar velocities in km/s
    
    return pos, vel # return the loaded particles or halo data

def load_data_halo(sim_path, obj, mass_limit=1):

    halo_all = obj.halo_selection(sim_path, '+', mass_limit)
    pos = halo_all[:, :3]
    vel = halo_all[:, 3:6]
    mass =  halo_all[:,6]
    return pos, vel, mass # return the loaded particles or halo data

def load_data_JD(bulk_species, sim_path, obj, mass_limit=1, ranknum=512):
    # particles read
    pos_data = []
    vel_data = []
    if bulk_species == 'cdm':
        # snapshot = sim_path  + '/' + sim + '/output/snap00' + str(snap_num) + '_cdm'
        for rank in range(ranknum):
            input_file = sim_path+"/0.000xv"+str(rank)+".dat"
            # input_file = "./../../simulations_JD/snapshots/0.000xv"+str(rank)+"_nu.dat"
            file_data = ReadParticleFile(input_file)
            # Split the data into components and append to separate lists
            pos_data_add = np.vstack((file_data[0], file_data[1], file_data[2])).T
            vel_data_add = np.vstack((file_data[3], file_data[4], file_data[5])).T
            pos_data.append(pos_data_add)
            vel_data.append(vel_data_add)
        pos_data = np.vstack(pos_data)
        vel_data = np.vstack(vel_data)
    elif bulk_species == 'nu':
        for rank in range(ranknum):
            input_file = sim_path+"/0.000xv"+str(rank)+"_nu.dat"
            # input_file = "./../../simulations_JD/snapshots/0.000xv"+str(rank)+"_nu.dat"
            file_data = ReadParticleFile(input_file)
            # Split the data into components and append to separate lists
            pos_data_add = np.vstack((file_data[0], file_data[1], file_data[2])).T
            vel_data_add = np.vstack((file_data[3], file_data[4], file_data[5])).T
            pos_data.append(pos_data_add)
            vel_data.append(vel_data_add)
        pos_data = np.vstack(pos_data)
        vel_data = np.vstack(vel_data)
    elif bulk_species == 'halo':
        mass_data = []
        for rank in range(ranknum):
            input_file = sim_path+"/0.000halo"+str(rank)+".dat"
            a = 1.;
            file_data = ReadHaloFile_data(input_file, a)
            cond = (file_data[6] >= mass_limit)  # Assuming file_data[6] contains the values you want to compare
            # Split the data into components and append to separate lists
            pos_data_add = np.vstack((file_data[0][cond], file_data[1][cond], file_data[2][cond])).T
            vel_data_add = np.vstack((file_data[3][cond], file_data[4][cond], file_data[5][cond])).T
            pos_data.append(pos_data_add)
            vel_data.append(vel_data_add)

        pos_data = np.vstack(pos_data)
        vel_data = np.vstack(vel_data) 
    return pos_data, vel_data # return the loaded particles or halo data


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
