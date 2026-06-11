import sys
import os
import time
import pickle
import numpy as np
import psutil
from mpi4py import MPI
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')
import MAS_library as MASL
import readsnap
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18')
from ReadHalos import *
from ReadHalos_dm import *
from ReadParticles import *


def bulk_calculation_pylians(boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, bulk_species, file_path, string, mass_limit, mass_width, save_path, type_data="normal"):
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
    if type_data == "normal":
        if bulk_species == "halo":
            check_file_exists_all_ranks(file_path)
        else:
            check_file_exists_all_ranks(file_path + ".0")
    elif type_data == "JD":
        if bulk_species == "halo":
            check_file_exists_all_ranks(os.path.join(file_path, "0.000halo0.dat"))
        elif bulk_species == "cdm":
            check_file_exists_all_ranks(os.path.join(file_path, "0.000xv0.dat"))
        elif bulk_species == "nu":
            check_file_exists_all_ranks(os.path.join(file_path, "0.000xv0_nu.dat"))
    # print_metadata(boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, string, mass_limit, mass_width, save_path)
    loop_ngrid_list(rank, boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max, ngrid_step, size, bulk_species, file_path, string, mass_limit, mass_width, save_path, type_data)
    comm.Barrier()
    print_total_usage(rank, start_time_all, start_mem_all)
    if rank == 0:
        print("\n*********** All bulk velocity computation finished! ***********\n", flush=True)
    comm.Barrier()  # Synchronize all processes

def loop_ngrid_list(rank, boxsize, num_pcl, sim_type, spec, ngrid_min, ngrid_max,
                    ngrid_step, size, bulk_species, file_path, string,
                    mass_limit, mass_width, save_path, type_data):

    boxsize = np.float64(boxsize)
    ngrid_list = list(range(ngrid_min, ngrid_max, ngrid_step))
    ngrid_list_split = np.array_split(ngrid_list, size)

    if len(ngrid_list_split[rank]) == 0:
        return

    halo_cache = None

    # Load halo catalogue once per rank for normal Rockstar halo catalogues.
    if type_data == "normal" and bulk_species == "halo":
        pos, vel, masses, counts = load_data_halo(
            file_path,
            string,
            mass_limit,
            mass_width=mass_width,
            Remove_subhalo="yes",
            return_counts=True,
        )

        halo_cache = {
            "pos": pos,
            "vel": vel,
            "masses": masses,
            "counts": counts,
        }
    if type_data == "JD" and bulk_species == "halo":
        pos, vel, masses = load_data_JD_all(
            bulk_species,
            file_path,
            sim_type,
            mass_limit=mass_limit,
            mass_width=mass_width,
            string=string,
        )
        halo_cache = {
            "pos": pos.astype(np.float32),
            "vel": vel.astype(np.float32),
            "masses": masses.astype(np.float32),
            "counts": {
                "num_halos": int(len(masses)),
                "total_halos_before_subhalo_cut": -1,
                "total_halos_after_subhalo_cut": int(len(masses)),
                "subhalo_removal_applied": False,
                "note": f"JD halo catalogues have no subhalo PID column; count is after mass selection string={string}.",
            },
        }

    for ngrid in ngrid_list_split[rank]:
        ngrid = int(ngrid)

        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        sub_box_data = {}

        print(type_data, bulk_species, flush=True)

        if type_data == "JD":
            if bulk_species == "halo":
                metadata = compute_metadata_from_halo_counts(
                    boxsize, num_pcl, sim_type, spec, ngrid,
                    ngrid_min, ngrid_max, ngrid_step, size,
                    bulk_species, file_path, string, mass_limit,
                    mass_width, save_path, halo_cache["counts"]
                )

                process_halo_arrays(
                    halo_cache["pos"],
                    halo_cache["vel"],
                    halo_cache["masses"],
                    ngrid,
                    boxsize,
                    sub_box_data
                )

            else:
                metadata = compute_metadata(
                    boxsize, num_pcl, sim_type, spec, ngrid,
                    ngrid_min, ngrid_max, ngrid_step, size,
                    bulk_species, file_path, string, mass_limit,
                    mass_width, save_path
                )

                process_cdm_nu_JD(
                    bulk_species, sim_type, file_path,
                    ngrid, boxsize, sub_box_data
                )

        elif type_data == "normal":
            if bulk_species == 'halo':
                metadata = compute_metadata_from_halo_counts(
                    boxsize, num_pcl, sim_type, spec, ngrid,
                    ngrid_min, ngrid_max, ngrid_step, size,
                    bulk_species, file_path, string, mass_limit,
                    mass_width, save_path, halo_cache["counts"]
                )

                process_halo_arrays(
                    halo_cache["pos"],
                    halo_cache["vel"],
                    halo_cache["masses"],
                    ngrid,
                    boxsize,
                    sub_box_data
                )

            else:
                metadata = compute_metadata(
                    boxsize, num_pcl, sim_type, spec, ngrid,
                    ngrid_min, ngrid_max, ngrid_step, size,
                    bulk_species, file_path, string, mass_limit,
                    mass_width, save_path
                )

                process_cdm_nu_data_gadget(
                    rank, file_path, bulk_species,
                    ngrid, boxsize, num_pcl, sub_box_data
                )

        else:
            raise ValueError(f"type_data must be 'normal' or 'JD', got {type_data!r}")

        simulation = simulation_def(
            boxsize, spec, num_pcl, sim_type,
            bulk_species, string, mass_limit, mass_width
        )

        save_and_print_usage(
            start_time, start_mem, simulation, ngrid,
            sub_box_data, bulk_species, metadata, save_path
        )

# Function to get MPI rank and size
def get_mpi_info():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def check_file_exists_all_ranks(path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        exists = os.path.exists(path)
    else:
        exists = None

    exists = comm.bcast(exists, root=0)

    if not exists:
        if rank == 0:
            print(f"Error: File not found: {path}", flush=True)
        comm.Abort(1)

# Function to check and print errors based on simulation type
def check_simulation_errors(sim_type, bulk_species, rank, comm):
    if sim_type == "0.0ev" and bulk_species == "nu" and rank == 0:
        print("In the case of LCDM we don't have nu snapshots!", flush=True)
        comm.Abort(1)


def compute_metadata(boxsize, Num_pcl_sim, sim_type, spec, ngrid,
                     ngrid_min, ngrid_max, ngrid_step, size,
                     bulk_species, file_path, string, mass_limit,
                     mass_width, save_path):

    if bulk_species == "halo":
        raise ValueError(
            "For halo metadata use compute_metadata_from_halo_counts(), "
            "to avoid reloading the halo catalogue."
        )

    meta_data = {
        'spec': spec,
        'boxsize': float(boxsize),
        'N_grids_simulation': Num_pcl_sim,
        'sim_type': sim_type,
        'ngrid': int(ngrid),
        'ngrid_min': ngrid_min,
        'ngrid_max': ngrid_max,
        'ngrid_step': ngrid_step,
        'file_path': file_path,
        'bulk_species': bulk_species,
        'save_path': save_path,
        'density': 'CIC-deposited particle count field: density(cell)=sum_i W_CIC(x_cell-x_i)',
        'P_x': 'CIC-deposited particle-number-weighted velocity field: P_x(cell)=sum_i W_CIC(x_cell-x_i) v_{x,i}; P_x/cell_volume is a number-current/momentum-density-like field for equal-mass particles; P_x/density is the mean velocity',
        'P_y': 'CIC-deposited particle-number-weighted velocity field: P_y(cell)=sum_i W_CIC(x_cell-x_i) v_{y,i}; P_y/cell_volume is a number-current/momentum-density-like field for equal-mass particles; P_y/density is the mean velocity',
        'P_z': 'CIC-deposited particle-number-weighted velocity field: P_z(cell)=sum_i W_CIC(x_cell-x_i) v_{z,i}; P_z/cell_volume is a number-current/momentum-density-like field for equal-mass particles; P_z/density is the mean velocity',
    }

    return meta_data

def compute_metadata_from_halo_counts(
    boxsize, Num_pcl_sim, sim_type, spec, ngrid,
    ngrid_min, ngrid_max, ngrid_step, size,
    bulk_species, file_path, string, mass_limit,
    mass_width, save_path, counts
):
    meta = {
        'spec': spec,
        'boxsize': float(boxsize),
        'N_grids_simulation': Num_pcl_sim,
        'sim_type': sim_type,
        'ngrid': int(ngrid),
        'ngrid_min': ngrid_min,
        'ngrid_max': ngrid_max,
        'ngrid_step': ngrid_step,
        'mass cut': "{:.4e}".format(mass_limit),
        'masses': string,
        'mass_width': mass_width,
        'file_path': file_path,
        'bulk_species': bulk_species,
        'save_path': save_path,
        'num_halos': counts["num_halos"],
        'total_halos_before_subhalo_cut': counts["total_halos_before_subhalo_cut"],
        'total_halos_after_subhalo_cut': counts["total_halos_after_subhalo_cut"],
        'subhalo_removal_applied': counts.get("subhalo_removal_applied", True),
        'note': counts.get("note", ""),
        'density': 'CIC-deposited halo count field: density(cell)=sum_h W_CIC(x_cell-x_h)',
        'P_x': 'CIC-deposited halo-number-weighted velocity field: P_x(cell)=sum_h W_CIC(x_cell-x_h) v_{x,h}; P_x/cell_volume is a halo number-current-like field; P_x/density is the mean halo velocity',
        'P_y': 'CIC-deposited halo-number-weighted velocity field: P_y(cell)=sum_h W_CIC(x_cell-x_h) v_{y,h}; P_y/cell_volume is a halo number-current-like field; P_y/density is the mean halo velocity',
        'P_z': 'CIC-deposited halo-number-weighted velocity field: P_z(cell)=sum_h W_CIC(x_cell-x_h) v_{z,h}; P_z/cell_volume is a halo number-current-like field; P_z/density is the mean halo velocity',
        'sum_b_M': 'CIC-deposited sum of halo weights w_h=b_h+(M_h/1.3e14)^0.85: sum_b_M(cell)=sum_h W_CIC w_h',
        'sum_b_M_vel_h_x': 'CIC-deposited weighted halo velocity field: sum_h W_CIC w_h v_{x,h}; divide by sum_b_M for weighted mean velocity',
        'sum_b_M_vel_h_y': 'CIC-deposited weighted halo velocity field: sum_h W_CIC w_h v_{y,h}; divide by sum_b_M for weighted mean velocity',
        'sum_b_M_vel_h_z': 'CIC-deposited weighted halo velocity field: sum_h W_CIC w_h v_{z,h}; divide by sum_b_M for weighted mean velocity',
    }
    return meta
# Function to process halo data
def process_halo_arrays(pos, vel, masses, ngrid, boxsize, sub_box_data):
    if masses.size == 0:
        raise ValueError("No halos selected. Check mass_limit, mass_width, and subhalo cut.")

    bias_h = np.float32(bias.haloBias(masses, model='sheth01', z=0.0, mdef='200m'))

    print(f"{pos.shape[0]} number of haloes loaded\n", flush=True)

    MAS = 'CIC'
    verbose = False

    rho = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, rho, boxsize, MAS, verbose=verbose)
    print("Density field is computed!\n", flush=True)

    bM_weight = bias_h + (masses / 1.3e14) ** 0.85

    sum_b_M_field = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_field, boxsize, MAS, W=bM_weight, verbose=verbose)
    print("sum_b_M_field field is computed!\n", flush=True)

    Vx = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, Vx, boxsize, MAS, W=vel[:, 0], verbose=verbose)

    Vy = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, Vy, boxsize, MAS, W=vel[:, 1], verbose=verbose)

    Vz = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, Vz, boxsize, MAS, W=vel[:, 2], verbose=verbose)

    sum_b_M_vel_h_x_field = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_vel_h_x_field, boxsize, MAS, W=vel[:, 0] * bM_weight, verbose=verbose)

    sum_b_M_vel_h_y_field = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_vel_h_y_field, boxsize, MAS, W=vel[:, 1] * bM_weight, verbose=verbose)

    sum_b_M_vel_h_z_field = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(pos, sum_b_M_vel_h_z_field, boxsize, MAS, W=vel[:, 2] * bM_weight, verbose=verbose)

    print("All fields are computed!\n", flush=True)

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

    snapshot = file_path
    header   = readsnap.snapshot_header(snapshot)
    Nall     = header.nall 
    # ptype    = header.format #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

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
# Deprecated: JD halos are now cached in loop_ngrid_list() and processed with process_halo_arrays().
# def process_data_JD_halos(bulk_species, sim_type, mass_limit, file_path, ngrid, boxsize, sub_box_data):
#     pos, vel, masses = load_data_JD_all(bulk_species, file_path, sim_type, mass_limit)
    
#     bias_h = np.float32(bias.haloBias(masses, model='sheth01', z=0.0, mdef='200m'))
#     print(f"{np.shape(pos)[0]} number of haloes loaded","\n")
#     axis     = 0       #no RSD
#     MAS      = 'CIC'   #it assumes the density constrast and velocities have been generated with the same MAS
#     threads  = 8       #number of openmp threads
#     verbose = False 
#     rho = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, rho, boxsize, MAS, verbose=verbose)
#     print("Density field is computed!","\n") # Note that the density is not normalized to the box, and to make the density we need to divide by L^3, it's basically number counts with CIC method

#     weight = (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
#     sum_b_M_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, sum_b_M_field, boxsize, MAS, W = weight, verbose=verbose)
#     print("sum_b_M_field field is computed!","\n")
#     ####
#     weight = vel[:,0]
#     Vx = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, Vx, boxsize, MAS, W = weight, verbose=verbose)

#     weight = vel[:,0] * (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
#     sum_b_M_vel_h_x_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, sum_b_M_vel_h_x_field, boxsize, MAS, W = weight, verbose=verbose)
#     #####
#     weight = vel[:,1]
#     Vy = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, Vy, boxsize, MAS, W = weight, verbose=verbose)

#     weight = vel[:,1] * (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
#     sum_b_M_vel_h_y_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, sum_b_M_vel_h_y_field, boxsize, MAS, W = weight, verbose=verbose)
    
#     #####
#     weight = vel[:,2]
#     Vz = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, Vz,boxsize, MAS, W = weight, verbose=verbose)

#     weight = vel[:,2] * (bias_h + (masses / (1.3 * 1.e14)) ** (0.85))
#     sum_b_M_vel_h_z_field = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
#     MASL.MA(pos, sum_b_M_vel_h_z_field, boxsize, MAS, W = weight, verbose=verbose)
    
#     print(f"All field are computed!","\n")

#     sub_box_data['density'] = rho
#     sub_box_data['sum_b_M'] = sum_b_M_field
#     sub_box_data['sum_b_M_vel_h_x'] = sum_b_M_vel_h_x_field
#     sub_box_data['sum_b_M_vel_h_y'] = sum_b_M_vel_h_y_field
#     sub_box_data['sum_b_M_vel_h_z'] = sum_b_M_vel_h_z_field
#     sub_box_data['P_x'] = Vx
#     sub_box_data['P_y'] = Vy
#     sub_box_data['P_z'] = Vz


# Function to process cdm/nu data
def process_cdm_nu_JD(bulk_species, sim_type, file_path, ngrid, boxsize, sub_box_data):

    pos, vel = load_data_JD_all(bulk_species, file_path, sim_type)
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
    print(f"Momentum field is computed! and loaded number of particles is {np.shape(pos)[0]}","\n")

    sub_box_data['density'] = rho
    sub_box_data['P_x'] = Vx
    sub_box_data['P_y'] = Vy
    sub_box_data['P_z'] = Vz



def simulation_def(boxsize, spec, N_pcl_sim, sim_type, species, string, mass_limit, mass_width):
    if species == "halo":
        if string == "+":
            return sim_type + '_' + spec + '_' + species + f'_mass_{mass_limit:.1e}'
        elif string == "bin":
            return sim_type + '_' + spec + '_' + species + f'_mass_{mass_limit:.1e}' + f'_mass_bin_{mass_width}'
        elif string == "all":
            return sim_type + '_' + spec + '_' + species + '_all'
        elif string == "-":
            return sim_type + '_' + spec + '_' + species + f'_mass_le_{mass_limit:.1e}'
        else:
            raise ValueError(f"Unknown halo mass-selection string={string!r}")
    else:
        return sim_type + '_' + spec + '_' + species
    
def load_data_gadget(sim_path, ptype):
    # particles read
    pos = readsnap.read_block(sim_path, "POS ", ptype)/1e3 #positions in Mpc/h #readsnap.read_block(snapshot, "POS ", ptype, True, 0, False, False,[0,num,0,0,0,0])/1e3
    vel = readsnap.read_block(sim_path, "VEL ", ptype)     #peculiar velocities in km/s
    return pos, vel # return the loaded particles or halo data

def load_data_halo(
    sim_path,
    string,
    mass_limit=1,
    mass_width=0.5,
    Remove_subhalo="yes",
    return_counts=False,
):
    result = halo_selection(
        sim_path,
        string,
        mass_limit,
        mass_width=mass_width,
        Remove_subhalo=Remove_subhalo,
        return_counts=return_counts,
    )

    if return_counts:
        halo_all, num_halos, total_before_subhalo_cut, total_after_subhalo_cut = result
    else:
        halo_all = result

    pos = halo_all[:, :3].astype(np.float32)
    vel = halo_all[:, 3:6].astype(np.float32)
    mass = halo_all[:, 6].astype(np.float32)

    if return_counts:
        counts = {
            "num_halos": int(num_halos),
            "total_halos_before_subhalo_cut": int(total_before_subhalo_cut),
            "total_halos_after_subhalo_cut": int(total_after_subhalo_cut),
        }
        return pos, vel, mass, counts

    return pos, vel, mass



def get_mass_selection_mask(masses, string, mass_limit, mass_width=0.5):
    """
    Return a boolean mask for mass selection.

    Parameters
    ----------
    masses : ndarray
        Halo masses.
    string : {"+", "-", "bin", "all"}
        Mass-selection mode.
    mass_limit : float
        Mass threshold or bin center.
    mass_width : float
        Bin-width parameter for string == "bin".
    """

    masses = np.asarray(masses)

    if string == "+":
        return masses > mass_limit

    elif string == "-":
        return masses <= mass_limit

    elif string == "all":
        return np.ones(masses.shape, dtype=bool)

    elif string == "bin":
        if mass_limit <= 0:
            raise ValueError("mass_limit must be positive for string='bin'.")

        exponent = np.floor(np.log10(mass_limit))
        coefficient = mass_limit / (10**exponent)

        if np.isclose(coefficient, 1.0):
            lower = (coefficient - 0.1 * mass_width) * 10**exponent
            upper = (coefficient + mass_width) * 10**exponent
        else:
            lower = (coefficient - mass_width) * 10**exponent
            upper = (coefficient + mass_width) * 10**exponent

        return (masses >= lower) & (masses <= upper)

    else:
        raise ValueError("string must be one of '+', '-', 'bin', or 'all'.")

def load_data_JD_all(
    bulk_species,
    sim_path,
    sim_type,
    mass_limit=1,
    mass_width=0.5,
    string="+",
    ranknum=512,):
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
        return pos_data, vel_data
        
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
        return pos_data, vel_data
        
    elif bulk_species == 'halo':

        mass_data = []
        if sim_type == "0.0ev":
            for rank in range(ranknum):
                input_file = sim_path+"/0.000halo"+str(rank)+".dat"
                a = 1.;
                file_data = ReadHaloFile_lcdm(input_file, a)
                cond = get_mass_selection_mask(file_data[6],string, mass_limit, mass_width=mass_width,)  # Assuming file_data[6] contains the values you want to compare
                # Split the data into components and append to separate lists
                pos_data_add = np.vstack((file_data[0][cond], file_data[1][cond], file_data[2][cond])).T
                vel_data_add = np.vstack((file_data[3][cond], file_data[4][cond], file_data[5][cond])).T
                pos_data.append(pos_data_add)
                vel_data.append(vel_data_add)
                mass_data.append(file_data[6][cond])
    
            pos_data = np.vstack(pos_data)
            vel_data = np.vstack(vel_data)
            mass_data = np.concatenate(mass_data)
            return pos_data, vel_data, mass_data
        
        else:    
            for rank in range(ranknum):
                input_file = sim_path+"/0.000halo"+str(rank)+".dat"
                a = 1.;
                file_data = ReadHaloFile_data(input_file, a)
                cond = get_mass_selection_mask(file_data[6],string, mass_limit, mass_width=mass_width,)  # Assuming file_data[6] contains the values you want to compare
                # Split the data into components and append to separate lists
                pos_data_add = np.vstack((file_data[0][cond], file_data[1][cond], file_data[2][cond])).T
                vel_data_add = np.vstack((file_data[3][cond], file_data[4][cond], file_data[5][cond])).T
                pos_data.append(pos_data_add)
                vel_data.append(vel_data_add)
                mass_data.append(file_data[6][cond])
    
            pos_data = np.vstack(pos_data)
            vel_data = np.vstack(vel_data)
            mass_data = np.concatenate(mass_data)
            return pos_data, vel_data, mass_data


def halo_selection(data_address, string, mass_limit, mass_width=0.5, Remove_subhalo="yes",  extra_columns=False, return_counts=False):
    """ 
    Filter a Rockstar halo catalogue.

    Parameters
    ----------
    data_address : str
        Path to the halo catalogue.
    string : {"+", "-", "bin", "all"}
        Selection mode:
        "+"   : select halos with M200b > mass_limit
        "-"   : select halos with M200b <= mass_limit
        "bin" : select halos in a mass bin around mass_limit
        "all" : select all halos
    mass_limit : float
        Mass threshold or bin center.
    mass_width : float, optional
        Width parameter for the mass bin.
    remove_subhalos : bool, optional
        If True, remove subhalos by requiring PID == -1.
        Assumes PID is the last column of a parent-processed catalogue.
    extra_columns : bool, optional
        If True, append Rvir as an extra column.
    return_counts : bool, optional
        If True, return halo_pop, num_halos, tot_halos.

    Returns
    -------
    halo_pop : ndarray
        Columns:
        [X, Y, Z, VX, VY, VZ, M200b]
        or
        [X, Y, Z, VX, VY, VZ, M200b, Rvir]
        if extra_columns=True.
    """
    data = np.loadtxt(data_address)
    data = np.atleast_2d(data)

    total_before_subhalo_cut = data.shape[0]

    if Remove_subhalo == "yes":
        if data.shape[1] < 42:
            raise ValueError(
                "Remove_subhalo='yes' requires a parent-processed Rockstar catalogue "
                "with PID as the last column. This file appears to have fewer than 42 columns."
            )

        pid = data[:, -1].astype(np.int64)

        if not np.any(pid == -1):
            raise ValueError(
                "Remove_subhalo='yes' but no PID == -1 entries found. "
                "Are you sure this is a parent-processed catalogue with PID as the last column?"
            )

        data = data[pid == -1]


    total_after_subhalo_cut = data.shape[0]

    condition = get_mass_selection_mask(
    data[:, 20],
    string,
    mass_limit,
    mass_width=mass_width,)

    selected = data[condition]

    if extra_columns:
        halo_pop = np.zeros((selected.shape[0], 8))
    else:
        halo_pop = np.zeros((selected.shape[0], 7))

    halo_pop[:, 0:6] = selected[:, 8:14]
    halo_pop[:, 6] = selected[:, 20]  # M200b

    if extra_columns:
        halo_pop[:, 7] = selected[:, 5]  # Rvir, kpc/h

    if return_counts:
        num_halos = halo_pop.shape[0]
        return halo_pop, num_halos, total_before_subhalo_cut, total_after_subhalo_cut

    return halo_pop



def save_dataframe(df, simulation, bulk_species, metadata, ngrid, save_path):
    # Add metadata
    # df.attrs['metadata'] = metadata
    data_to_save = {
        'metadata': metadata,
        'sub_box_data': df
    }

    # Save the dataframe
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(
        save_path,
        f"data_ngrid_{ngrid}_sim_{simulation}.pickle")

    with open(output_file, 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
