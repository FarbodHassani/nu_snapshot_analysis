import sys
import numpy as np
import pickle
import os
import h5py
import json
import time
import psutil
import pickle
import pandas as pd
from mpi4py import MPI
import re
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')
import MAS_library as MASL
import readgadget
import readsnap
from collections import defaultdict
from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from ReadHalos import *
from ReadHalos_dm import *
from ReadParticles import *


def bulk_calculation_pylians(boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, bulk_species, file_path, mass_limit, save_path, type_data="normal"):
    """
    Perform bulk calculation over the specified parameters.

    Args:
        boxsize (float): Size of the simulation box.
        Num_pcl_sim (int): Number of particles in the simulation.
        sim_type (str): Type of simulation ('JD', 'normal', etc.).
        ngrid_min (int): Minimum value of grid size.
        ngrid_max (int): Maximum value of grid size.
        ngrid_step (int): Step size for grid size.
        size (int): Number of MPI processes.
        bulk_species (str): Type of bulk species (e.g., 'cdm', 'halo', 'nu').
        file_path (str): Path to the simulation files.
        mass_limit (float): Mass limit for filtering particles.
        save_path (str): Path to save the results.
        type_data (str, optional): Type of data processing ('normal' or 'JD'). Defaults to 'normal'.
    """
    boxsize = np.float64(boxsize)
    start_time_all = time.time()
    start_mem_all = psutil.Process().memory_info().rss
    comm, rank, size = get_mpi_info()
    check_simulation_errors(sim_type, bulk_species, rank, comm)
    print_metadata(boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path)
    loop_ngrid_list(rank, boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path, type_data)
    comm.Barrier()
    print_total_usage(rank, start_time_all, start_mem_all)
    if rank == 0:
        print("\n*********** All bulk velocity computation finished! ***********\n", flush=True)
    comm.Barrier()  # Synchronize all processes

# Function to loop over ngrid_list and process data
def loop_ngrid_list(rank, boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path, type_data):
    """
    Loop over the list of grid sizes and process data accordingly.

    Args:
        rank (int): Rank of the MPI process.
        ngrid_min (int): Minimum value of grid size.
        ngrid_max (int): Maximum value of grid size.
        ngrid_step (int): Step size for grid size.
        size (int): Total number of MPI processes.
        bulk_species (str): Type of bulk species (e.g., 'cdm', 'halo', 'nu').
        file_path (str): Path to the simulation files.
        mass_limit (float): Mass limit for filtering particles.
        save_path (str): Path to save the results.
        type_data (str, optional): Type of data processing ('normal' or 'JD'). Defaults to 'normal'.
    """
    boxsize = np.float64(boxsize);
    ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
    ngrid_list_split = np.array_split(ngrid_list, size)
    if len(ngrid_list_split[rank]) == 0:
        return  # Skip this rank if no grid points are assigned
    for ngrid in ngrid_list_split[rank]:
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        metadata = compute_metadata(boxsize, num_pcl, sim_type, spec, ngrid, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path)
        sub_box_data = {}
        if type_data == "JD":
            process_data_JD(rank, bulk_species, sim_type, mass_limit, file_path, ngrid, boxsize, sub_box_data)
        elif type_data == "normal":
            if bulk_species == 'halo':
                process_halo_data(rank, file_path, mass_limit, ngrid, boxsize, sub_box_data)
            else:
                process_cdm_nu_data_gadget(rank, file_path, bulk_species, ngrid, boxsize, num_pcl, sub_box_data)
        simulation = simulation_def(boxsize, spec, num_pcl, sim_type, bulk_species, mass_limit)
        save_and_print_usage(start_time, start_mem, simulation, ngrid, sub_box_data, bulk_species, metadata, save_path)


# Function to get MPI rank and size
def get_mpi_info():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

# Function to check and print errors based on simulation type
def check_simulation_errors(sim_type, bulk_species, rank, comm):
    if sim_type == "0.0ev" and bulk_species == "nu" and rank == 0:
        print("In the case of LCDM we don't have nu snapshots!", flush=True)
        comm.Abort(1)

# Function to print metadata
def print_metadata(boxsize, Num_pcl_sim, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path):
    meta_data = {'spec': spec, 'boxsize': boxsize, 'N_grids_simulation': Num_pcl_sim, 'sim_type' :sim_type, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step, 'mass cut = ':"{:.4e}".format(mass_limit)
                , 'file_path': file_path, 'bulk_species': bulk_species, 'save_path': save_path, 'sum_b_M':'<(bias_h + (mass)_h/(1.3e14))^0.85> average in each sub-box','sum_b_M_vel_h':'<(bias_h + (mass)_h/(1.3e14))^0.85 * v_h> average in each sub-box'};
    # print(meta_data)

def compute_metadata(boxsize, Num_pcl_sim, sim_type, spec, ngrid, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path):

    if bulk_species == 'halo':
        meta_data = {'spec': spec, 'boxsize': boxsize, 'N_grids_simulation': Num_pcl_sim, 'sim_type' :sim_type, 'ngrid': ngrid, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step, 'mass cut = ':"{:.4e}".format(mass_limit)
                , 'file_path': file_path, 'bulk_species': bulk_species, 'ngrid': ngrid, 'save_path': save_path, 'sum_b_M':'<(bias_h + (mass)_h/(1.3e14))^0.85> average in each sub-box','sum_b_M_vel_h':'<(bias_h + (mass)_h/(1.3e14))^0.85 * v_h> average in each sub-box'};
    else:
        meta_data = {'spec': spec, 'boxsize': boxsize, 'N_grids_simulation': Num_pcl_sim, 'sim_type' :sim_type, 'ngrid': ngrid, 'ngrid_min': ngrid_min, 'ngrid_max': ngrid_max, 'ngrid_step': ngrid_step, 'file_path': file_path, 'bulk_species': bulk_species, 'ngrid': ngrid, 'save_path': save_path};

    return meta_data

# Function to process halo data
def process_halo_data(rank, file_path, mass_limit, ngrid, boxsize, sub_box_data):
    if rank == 0:
        file_exists = os.path.exists(file_path)
    else:
        file_exists = None
    
    # Broadcast the result of the check to all processes
    comm = MPI.COMM_WORLD
    file_exists = comm.bcast(file_exists, root=0)

    # If the file doesn't exist, print error and abort
    if not file_exists:
        # if rank == 0:
        print(f"Error: File not found at {file_path}. Exiting all processes.", flush=True)
        comm.Barrier()  # Ensure all ranks wait before aborting
        comm.Abort(1)
    pos, vel, masses = load_data_halo(file_path, mass_limit)
    bias_h = np.float32(bias.haloBias(masses, model='sheth01', z=0.0, mdef='200m'))
    print(f"{np.shape(pos)[0]} number of haloes loaded","\n")
    axis     = 0       #no RSD
    MAS      = 'CIC'   #it assumes the density constrast and velocities have been generated with the same MAS
    threads  = 8       #number of openmp threads
    verbose = False 
    rho = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, rho, boxsize, MAS, verbose=verbose)
    print("Density field is computed!","\n") # Note that the density is not normalized to the box, and to make the density we need to divide by L^3, it's basically number counts with CIC method

    weight = (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
    sum_b_M_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_field, boxsize, MAS, W = weight, verbose=verbose)
    print("sum_b_M_field field is computed!","\n")
    ####
    weight = vel[:,0]
    Vx = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, Vx, boxsize, MAS, W = weight, verbose=verbose)

    weight = vel[:,0] * (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
    sum_b_M_vel_h_x_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_vel_h_x_field, boxsize, MAS, W = weight, verbose=verbose)
    #####
    weight = vel[:,1]
    Vy = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, Vy, boxsize, MAS, W = weight, verbose=verbose)

    weight = vel[:,1] * (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
    sum_b_M_vel_h_y_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_vel_h_y_field, boxsize, MAS, W = weight, verbose=verbose)
    
    #####
    weight = vel[:,2]
    Vz = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, Vz,boxsize, MAS, W = weight, verbose=verbose)

    weight = vel[:,1] * (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
    sum_b_M_vel_h_z_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_vel_h_z_field, boxsize, MAS, W = weight, verbose=verbose)
    
    print(f"All field are computed!","\n")

    sub_box_data['density'] = rho
    sub_box_data['sum_b_M'] = sum_b_M_field
    sub_box_data['sum_b_M_vel_h_x'] = sum_b_M_vel_h_x_field
    sub_box_data['sum_b_M_vel_h_y'] = sum_b_M_vel_h_y_field
    sub_box_data['sum_b_M_vel_h_z'] = sum_b_M_vel_h_z_field
    sub_box_data['P_x'] = Vx
    sub_box_data['P_y'] = Vy
    sub_box_data['P_z'] = Vz


def save_and_print_usage(start_time, start_mem, simulation, ngrid, sub_box_data, bulk_species, metadata, save_path):
    save_dataframe(sub_box_data, simulation, bulk_species, metadata, ngrid, save_path)
    print(f"{simulation}\n", flush=True)
    print_usage(start_time, start_mem, f" n_grid={ngrid} Finished!\n")


def print_usage(start_time, start_mem, message=""):
    end_time = time.time()
    end_mem = psutil.Process().memory_info().rss
    elapsed_time = end_time - start_time
    elapsed_mem = end_mem - start_mem
    print(f"{message} - Time: {elapsed_time:.2f} s, Memory: {elapsed_mem / 1024 / 1024:.2f} MB", flush=True)


# Function to synchronize all processes
def synchronize_processes(comm):
    comm.Barrier()

# Function to print total time and memory usage
def print_total_usage(rank, start_time_all, start_mem_all):
    if rank == 0:
        print_usage(start_time_all, start_mem_all, '- Total time and memory!')

# Function to process cdm/nu data
def process_cdm_nu_data_gadget(rank, file_path, bulk_species, ngrid, boxsize, num_pcl, sub_box_data):

    if rank == 0:
        file_exists = os.path.exists(file_path+".0")
    else:
        file_exists = None
    
    # Broadcast the result of the check to all processes
    comm = MPI.COMM_WORLD
    file_exists = comm.bcast(file_exists, root=0)

    # If the file doesn't exist, print error and abort
    if not file_exists:
        # if rank == 0:
        print(f"Error: File not found at {file_path}. Exiting all processes.", flush=True)
        comm.Barrier()  # Ensure all ranks wait before aborting
        comm.Abort(1)

    snapshot = file_path
    header   = readsnap.snapshot_header(snapshot)
    Nall     = header.nall 
    ptype    = header.format #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    if np.all(Nall == 0) or num_pcl >= 2048:
        if rank == 0:
            print(f"Nall in gadget 2 fromat is 0 and we are using number of particles explicitely to read the file through readsnap.read_block", flush=True)
        Nall = [0,num_pcl**3,0,0,0,0]
        pos = readsnap.read_block(snapshot, "POS ", 1, True,  0, False, False, Nall)/1000.  # Internal unit is Kpc/h and we should convert to Mpc/h
        vel = readsnap.read_block(snapshot, "VEL ", 1, True,  0, False, False, Nall)
    else:
        if rank == 0:
            print(f"Nall in gadget 2 fromat is non zero and we are using readgadget.read_block function", flush=True)
        pos = readsnap.read_block(snapshot, "POS ", 1, True,  0, False, False, Nall)/1000. # Internal unit is Kpc/h and we should convert to Mpc/h
        vel = readsnap.read_block(snapshot, "VEL ", 1, True,  0, False, False, Nall)
    print(f"{np.shape(pos)[0]} number of particles loaded which should be consistent with {Nall} from gadget 2 header or the third power of {num_pcl}","\n")
    axis     = 0       #no RSD
    MAS      = 'CIC'   #it assumes the density constrast and velocities have been generated with the same MAS
    threads  = 8       #number of openmp threads
    verbose = False 
    rho = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    # print("array hosting density field is defined!","\n")
    MASL.MA(pos, rho, boxsize, MAS, verbose=verbose)
    print("Density field is computed!","\n")
    weight = vel[:,0]
    Vx = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, Vx, boxsize,MAS, W = weight, verbose=verbose)
    
    weight = vel[:,1]
    Vy = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, Vy, boxsize,MAS, W = weight, verbose=verbose)
    
    weight = vel[:,2]
    Vz = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    MASL.MA(pos, Vz,boxsize, MAS, W = weight, verbose=verbose)
    print(f"Momentum field is computed! and loaded number of particles is {np.shape(pos)[0]} which should be consistent with {Nall}","\n")

    sub_box_data['density'] = rho
    sub_box_data['P_x'] = Vx
    sub_box_data['P_y'] = Vy
    sub_box_data['P_z'] = Vz

# Function to load JD simulation data
def process_data_JD(rank, bulk_species, sim_type, mass_limit, file_path, ngrid, boxsize, sub_box_data):

    # from ReadHalos import ReadHaloFile_lcdm, ReadHaloFile_data
    # from ReadParticles import ReadParticleFile
    rank_total = 512;
    for rank_id in range(rank_total):
        if bulk_species == 'halo':
            if sim_type == "0.0ev":
                input_file = file_path + "/0.000halo" + str(rank_id) + ".dat"
                a = 1.
                file_data = ReadHaloFile_lcdm(input_file, a)  # Assuming ReadHaloFile_lcdm is the function to read JD data
                cond = (file_data[6] >= mass_limit)
                masses = file_data[6][cond]
                pos = np.vstack((file_data[0][cond], file_data[1][cond], file_data[2][cond])).T
                vel = np.vstack((file_data[3][cond], file_data[4][cond], file_data[5][cond])).T
            else:
                input_file = file_path + "/0.000halo" + str(rank_id) + ".dat"
                a = 1.
                file_data = ReadHaloFile_data(input_file, a)  # Assuming ReadHaloFile_data is the function to read JD data
                cond = (file_data[6] >= mass_limit)
                masses = file_data[6][cond]
                pos = np.vstack((file_data[0][cond], file_data[1][cond], file_data[2][cond])).T
                vel = np.vstack((file_data[3][cond], file_data[4][cond], file_data[5][cond])).T
            for pcl in range(np.shape(pos)[0]):
                x_index = int(np.floor( (pos[pcl, 0]/boxsize) * ngrid ))
                y_index = int(np.floor( (pos[pcl, 1]/boxsize) * ngrid ))
                z_index = int(np.floor( (pos[pcl, 2]/boxsize) * ngrid ))
                sub_box_index = (x_index, y_index, z_index)
                
                mass = masses[pcl]
                bias_h = bias.haloBias(mass, model='sheth01', z=0.0, mdef='200m')
                if sub_box_index not in sub_box_data:
                    sub_box_data[sub_box_index] = {
                        'sum_bulk_vel': 0.0,
                        'sum_b_M': 0.0,
                        'sum_b_M_vel_h': 0.0,
                        'N': 0
                    }
                sub_box_data[sub_box_index]['sum_bulk_vel'] += vel[pcl]
                sub_box_data[sub_box_index]['sum_b_M'] += (bias_h + (mass / (1.3 * 1.e14)) ** (0.85))
                sub_box_data[sub_box_index]['sum_b_M_vel_h'] += (bias_h + (mass / (1.3 * 1.e14)) ** (0.85)) * vel[pcl]
                sub_box_data[sub_box_index]['N'] += 1
        else:
            if bulk_species == 'cdm':
                input_file = file_path + "/0.000xv" + str(rank_id) + ".dat"
            elif bulk_species == 'nu':
                input_file = file_path + "/0.000xv" + str(rank_id) + "_nu.dat"
            file_data = ReadParticleFile(input_file)  # Assuming ReadParticleFile is the function to read JD data
            pos = np.vstack((file_data[0], file_data[1], file_data[2])).T
            vel = np.vstack((file_data[3], file_data[4], file_data[5])).T
            for pcl in range(np.shape(pos)[0]):
                x_index = int(np.floor( (pos[pcl, 0]/boxsize) * ngrid ))
                y_index = int(np.floor( (pos[pcl, 1]/boxsize) * ngrid ))
                z_index = int(np.floor( (pos[pcl, 2]/boxsize) * ngrid ))
                sub_box_index = (x_index, y_index, z_index)
                if sub_box_index not in sub_box_data:
                    sub_box_data[sub_box_index] = {
                        'sum_bulk_vel': 0.0,
                        'N': 0
                    }
                sub_box_data[sub_box_index]['sum_bulk_vel'] += vel[pcl]
                sub_box_data[sub_box_index]['N'] += 1


def simulation_def(boxsize, spec, N_pcl_sim, sim_type, species, mass_limit):
    if species == "halo":
        return sim_type+'_'+spec+'_'+species+f'_mass_{mass_limit:.1e}'
    else:
        return sim_type+'_'+spec+'_'+species
    
def load_data_gadget(sim_path, ptype):
    # particles read
    pos = readsnap.read_block(sim_path, "POS ", ptype)/1e3 #positions in Mpc/h #readsnap.read_block(snapshot, "POS ", ptype, True, 0, False, False,[0,num,0,0,0,0])/1e3
    vel = readsnap.read_block(sim_path, "VEL ", ptype)     #peculiar velocities in km/s
    return pos, vel # return the loaded particles or halo data

def load_data_halo(sim_path, mass_limit=1):
    halo_all = halo_selection(sim_path, '+', mass_limit)
    pos = halo_all[:, :3]
    vel = halo_all[:, 3:6]
    mass =  halo_all[:,6]
    return pos.astype(np.float32), vel.astype(np.float32), mass.astype(np.float32) # return the loaded particles or halo data

def load_data_JD_all(bulk_species, sim_path, mass_limit=1, ranknum=2):
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


def halo_selection(data_address, string, mass_limit, Remove_subhalo="no",  extra_columns=False):
    """ 
    This function filters a halo catalogue from Rockstar based on a given mass limit and a string indicating whether to select larger or smaller halos.
Input:
- data_address: file path to the halo catalogue
- string: "+" for selecting halos above the mass limit, "-" for selecting halos below the mass limit, "all" for selecting all halos
- mass_limit: the mass limit for selecting halos
- extra_columns (default=False): flag for whether to include additional columns of [Rvir] in the output
Output:
- A new catalogue with [x,y,z,v_x,v_y,v_z, M200b] and additional columns of [Rvir] of halo population if requested
    """
    data = np.loadtxt(data_address);
    
    if (Remove_subhalo=='yes'):
        data = data[data[:,41]==-1]; # Only choose parent halos
    if (string == "+"):
        condition = data[:,20]> mass_limit; # M200b
        if extra_columns == True:
            halo_pop = np.zeros((np.shape(data[condition])[0],8))
        else:
            halo_pop = np.zeros((np.shape(data[condition])[0],7))
        for i in range(6):
            halo_pop[:,i] = data[condition][:,8+i]
        halo_pop[:,6] = data[condition][:,20] # Mass 

        if extra_columns == True:
            halo_pop[:,7] = data[condition][:,5] # R_vir
            
    elif (string == "-"):
        condition = data[:,20]<= mass_limit;
        if extra_columns == True:
            halo_pop = np.zeros((np.shape(data[condition])[0],8))
        else:
            halo_pop = np.zeros((np.shape(data[condition])[0],7))
            
        for i in range(6):
            halo_pop[:,i] = data[condition][:,8+i]
        halo_pop[:,6] = data[:,20] # Mass 200b

        if (extra_columns):
            halo_pop[:,7] = data[condition][:,5] # R_vir

    elif (string == "all"):
        if extra_columns == True:
            halo_pop = np.zeros((np.shape(data)[0],8))
        else:
            halo_pop = np.zeros((np.shape(data)[0],7))
        for i in range(6):
            halo_pop[:,i] = data[:,8+i]
        halo_pop[:,6] = data[:,20] # Mass 200b
        
        if (extra_columns):
            halo_pop[:,7] = data[:,5] # R_vir
            # halo_pop[:,7] = data[:,20] # Mass
    return halo_pop;


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

