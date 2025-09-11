##### In this library we plan to compute the spectra for density and velocity for the given bulk velocity information
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
import Pk_library as PKL
from collections import defaultdict
from analysis_functions import load



def power_spectra_pylians(boxsize, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, bulk_species, file_path, mass_limit, save_path):
    """
    Perform bulk calculation over the specified parameters.

    Args:
        boxsize (float): Size of the simulation box.
        sim_type (str): Type of simulation ('JD', 'normal', etc.).
        ngrid_min (int): Minimum value of grid size.
        ngrid_max (int): Maximum value of grid size.
        ngrid_step (int): Step size for grid size.
        size (int): Number of MPI processes.
        bulk_species (str): Type of bulk species (e.g., 'cdm', 'halo', 'nu').
        file_path (str): Path to the simulation files.
        mass_limit (float): Mass limit for filtering particles.
        save_path (str): Path to save the results.
    """
    boxsize = np.float64(boxsize)
    start_time_all = time.time()
    start_mem_all = psutil.Process().memory_info().rss
    comm, rank, size = get_mpi_info()
    check_simulation_errors(sim_type, bulk_species, rank, comm)
    loop_spectra_computation(rank, boxsize, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path)
    comm.Barrier()
    # print(f"Rank {rank}: passed first barrier", flush=True)
    
    print_total_usage(rank, start_time_all, start_mem_all)
    
    comm.Barrier()
    if rank == 0:
        print("\n*********** All spectra computation finished! ***********\n", flush=True)
    
    # comm.Barrier()
    # print(f"Rank {rank}: exiting program", flush=True)

# Function to loop over ngrid_list and process data
def loop_spectra_computation(rank, boxsize, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, mass_limit, save_path):
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
    """
    boxsize = np.float64(boxsize);
    ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
    ngrid_list_split = np.array_split(ngrid_list, size)
    if len(ngrid_list_split[rank]) == 0:
        return  # Skip this rank if no grid points are assigned
    for ngrid in ngrid_list_split[rank]:
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        power_data = {}
        compute_spectra(rank, file_path, bulk_species, spec, ngrid, sim_type, boxsize, power_data, mass_limit)
        simulation = simulation_def(boxsize, spec, sim_type, bulk_species, mass_limit)
        save_and_print_usage(start_time, start_mem, simulation, ngrid, power_data, save_path)

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


# Function to process halo data
def compute_spectra(rank, file_path, bulk_species, spec, ngrid, sim, boxsize, power_data, mass_cut = 1.0):
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
        
    data = load_files_pickles(file_path, bulk_species, spec, ngrid, sim, mass_cut)
    power_data['metadata'] = data['metadata']
    power_data['metadata']['info_spectra_rho_x_p'] = [" compute the density constrast and momentum auto- and cross-power spectra, k will have units of h/Mpc \
    Pk_dd will contain the standard power spectrum in (Mpc/h)^3 \
    Pk_tt will be the momentum auto-power spectrum, defined as above, and with units of (km/s)^2*(Mpc/h)^3 in units of velocity field are (km/s) \
    Pk_dt will be the density-momentum cross-power spectrum with (km/s)*(Mpc/h)^3 units if velocity field has (km/s) units"]
    
    power_data['metadata']['spectra_theta'] = ["k, Pk, Nmodes: compute the theta auto-power spectrum  k will be in h/Mpc units. Pk will have (km/s)^2*(Mpc/h)^3 considering that the velocity field is in km/s"]
    
    density = data['sub_box_data']['density']
    # Avoid division by zero by setting a small density value
    min_density = 1e-25  # Small positive value
    density = np.where(density <= 0, min_density, density)
    delta = density / np.mean(density, dtype=np.float64) - 1.0
    Vx = data['sub_box_data']['P_x']
    Vy = data['sub_box_data']['P_y']
    Vz = data['sub_box_data']['P_z']
    # Normalize velocity by density
    Vx = np.where(density > min_density, Vx / density, 0)
    Vy = np.where(density > min_density, Vy / density, 0)
    Vz = np.where(density > min_density, Vz / density, 0)
    # Compute spectra
    axis     = 0       #no RSD
    MAS      = 'CIC'   #it assumes the density constrast and velocities have been generated with the same MAS
    threads  = 1       #number of openmp threads
    verbose = False 
    ####
    # compute the density constrast and momentum auto- and cross-power spectra
    # k will have units of h/Mpc
    # Pk_dd will contain the standard power spectrum in (Mpc/h)^3
    # Pk_tt will be the momentum auto-power spectrum, defined as above, and with units of (km/s)^2*(Mpc/h)^3 in units of velocity field are (km/s)
    # Pk_dt will be the density-momentum cross-power spectrum with (km/s)*(Mpc/h)^3 units if velocity field has (km/s) units
    k, Pk_dd0, Pk_tt0, Pk_dt0, Nmodes = PKL.XPk_dv(delta, Vx, Vy, Vz, boxsize, axis, MAS, threads)
    power_data['spectra_rho_x_p'] = [k, Pk_dd0, Pk_tt0, Pk_dt0, Nmodes]

    # compute the theta auto-power spectrum
    k, Pk, Nmodes = PKL.Pk_theta(Vx,Vy,Vz, boxsize,axis,MAS,threads)
    power_data['spectra_theta'] = [k, Pk, Nmodes]  # k will be in h/Mpc units. Pk will have (km/s)^2*(Mpc/h)^3 considering that the velocity field is in km/s
   
    

def save_and_print_usage(start_time, start_mem, simulation, ngrid, power_data, save_path):
    save_dataframe(power_data, simulation, ngrid, save_path)
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

def simulation_def(boxsize, spec, sim_type, species, mass_limit):
    if species == "halo":
        return sim_type+'_'+spec+'_'+species+f'_mass_{mass_limit:.1e}'
    else:
        return sim_type+'_'+spec+'_'+species
    
def load_files_pickles(file_path, bulk_species, spec, ngrid, sim, mass_cut = 1.0):
    # data read
    if bulk_species == "halo":
        file_path = f"{file_path}/{spec}/{sim}/"+f"halo_mass_{mass_cut:.0e}".replace('+', '')+f"/output//data_ngrid_{ngrid}_sim_{sim}_{spec}_halo_mass_{mass_cut:.1e}.pickle"
    else:
        file_path = f"{file_path}/{spec}/{sim}/{bulk_species}/output//data_ngrid_{ngrid}_sim_{sim}_{spec}_{bulk_species}.pickle"
        
    data = load(file_path)
    return data # return the loaded particles or halo data



def save_dataframe(power_data, simulation, ngrid, save_path):

    # Save the dataframe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path+'/Spectra_ngrid_'+str(ngrid)+'_sim_'+simulation+'.pickle', 'wb') as handle:
        pickle.dump(power_data, handle)

