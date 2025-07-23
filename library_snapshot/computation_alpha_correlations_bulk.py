import numpy as np
import pandas as pd
import pickle
from mpi4py import MPI
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import dill
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')

## TODO: Important! Note that in the original method of beta, we need to compute x_i - <x>, where <x> means the average over all halos, which is not considered properly here! Need to be improved!

def nested_dict(n, type): # definingnested dictionary!
    if n==1:
        defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def perform_analysis_alpha_correlation(sims, specs, Mass_cuts, boxsize, save_path, ngrid_list, n_h_threshold = 3, num_cores=1):
    """
    Main function to perform analysis for different combinations of simulation parameters.
    Parameters:
    - sims (list): List of simulation identifiers.
    - specs (list): List of specification identifiers.
    - Mass_cuts (list): List of mass cut values.
    - boxsize (float): Box size of the simulation.
    - save_path (str): Path to save the analyzed data.
    - ngrid_list (list): List of grid points to analyze.
    - num_cores (int): Number of cores to be used for parallel processing.
    """

    #################
    ## MPI part
    #################
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    boxsize = np.float64(boxsize);
    if rank == 0:
        total_combinations = len(Mass_cuts) * len(specs) * len(sims)

        # Check if the number of cores matches the required number
        if num_cores < 1:
            print("Error: Invalid number of cores specified.")
            return
        elif num_cores > size:
            print(f"Error: Number of specified cores ({num_cores}) is greater than the available MPI processes ({size}).")
            return

        # Calculate the number of combinations each process should handle
        combinations_per_process = total_combinations // num_cores
        remainder = total_combinations % num_cores

        # Iterate over combinations assigned to each process
        for process_index in range(1, num_cores):
            start_index = process_index * combinations_per_process + min(process_index, remainder)
            end_index = (process_index + 1) * combinations_per_process + min(process_index + 1, remainder)
            comm.send((start_index, end_index), dest=process_index)

        # Calculate combinations for rank 0
        start_index = 0
        end_index = combinations_per_process + min(1, remainder)
    else:
        start_index, end_index = comm.recv(source=0)

    # Iterate over combinations assigned to this process
    for index in range(start_index, end_index):
        # Calculate the corresponding Mass_cut, spec, and sim
        Mass_cut_index = index % len(Mass_cuts)
        spec_index = (index // len(Mass_cuts)) % len(specs)
        sim_index = index // (len(Mass_cuts) * len(specs))
        Mass_cut = Mass_cuts[Mass_cut_index]
        spec = specs[spec_index]
        sim = sims[sim_index]
        analyze_and_save_for_ngrid(spec, sim, Mass_cut, ngrid_list, n_h_threshold, save_path, boxsize)

    # Wait for all processes to complete
    comm.Barrier()
    if rank == 0:
        print("Perform analysis finished!")


def load_data_bulk(file_path, ngrid, sim, spec, Mass_cut):  ### Loading the bulk velocities
    """
    Load data.
    Parameters:
    - file_path (str): Path to the data files.
    - ngrid (int): Number of grid points.
    - sim (str): Simulation identifier.
    - spec (str): Specification identifier.
    - Mass_cut (float): Mass cut value.
    Returns:
    - cdm_bulk_all, nu_bulk_all, halo_bulk_all: Loaded data.
    - Data based on analysis selection.

    """
    cdm_analysis = True; nu_analysis = True;
    
    if spec=="L_500_Ngrid_6144":
        halo_all_path = f"{file_path}/{convert_mass_cuts([Mass_cut])[0]}/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_halo_mass_{Mass_cut:.1e}.pickle"
        # sim_type+'_L_'+str(boxsize)+'_Ngrid_'+str(N_pcl_sim)+'_'+species+f'_mass_{mass_limit:.1e}'
        cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_cdm.pickle"
        nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_nu.pickle" if sim != "0.0ev" else ""
    # else:
    #     # cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_cdm.pickle"
    #     # nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_ncdm0.pickle" if sim != "0.0ev" else ""
    #     cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_cdm.pickle"
    #     nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_nu.pickle" if sim != "0.0ev" else ""
    #     halo_all_path = f"{file_path}/{convert_mass_cuts([Mass_cut])[0]}/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_halo_mass_{Mass_cut:.1e}.pickle"
    # for path in [cdm_path, nu_path] if cdm_analysis or nu_analysis else []:
    #     if path and not os.path.exists(path):
    #         print(f"Warning: {path} doesn't exist.")
    else:
        halo_all_path = f"{file_path}/{convert_mass_cuts([Mass_cut])[0]}/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_halo_mass_{Mass_cut:.1e}.pickle"

        # Preferred paths
        cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_cdm.pickle"
        nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_nu.pickle" if sim != "0.0ev" else ""

        # Check if preferred paths exist, otherwise use alternative paths
        if not os.path.exists(cdm_path):
            cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_cdm.pickle"

        if nu_path and not os.path.exists(nu_path):
            nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_ncdm0.pickle" if sim != "0.0ev" else ""

    # Check existence of paths and print warnings if they don't exist
    for path in [cdm_path, nu_path] if cdm_analysis or nu_analysis else []:
        if path and not os.path.exists(path):
            print(f"Warning: {path} doesn't exist.")
            
    # Load the data if the files exist
    cdm_bulk_all = np.load(cdm_path, allow_pickle=True) if cdm_analysis and os.path.exists(cdm_path) else None
    nu_bulk_all = np.load(nu_path, allow_pickle=True) if nu_analysis and os.path.exists(nu_path) and sim != "0.0ev" else None
    halo_bulk_all = np.load(halo_all_path, allow_pickle=True)

    return cdm_bulk_all, nu_bulk_all, halo_bulk_all

def convert_mass_cuts(Mass_cuts):
    """
    Convert mass cuts into variable names.

    Parameters:
    - Mass_cuts (list): List of mass cuts.

    Returns:
    - halo_masses (list): List of variable names for each mass cut.
    """
    halo_masses = []
    for mass_cut in Mass_cuts:
        notation = "{:.0e}".format(mass_cut)
        power, exponent = notation.split('e')
        if exponent.startswith('+'):
            exponent = exponent[1:]  # Remove the leading '+' symbol
        var_name = f"halo_mass_{power}e{exponent}"
        halo_masses.append(var_name)
    return halo_masses


def compute_dot_products(neutrino_velocity, halo_velocity, cdm_velocity):
    """
    This function computes various dot products and denominators for given velocity vectors
    of neutrinos, halos, and CDM. These calculations are performed for each sub-box separately.

    Parameters:
    - neutrino_velocity: Velocity vector of neutrinos in the sub-box.
    - halo_velocity: Velocity vector of halos in the sub-box.
    - cdm_velocity: Velocity vector of CDM in the sub-box.

    Returns:
    - An array containing the computed dot products and denominators.
    """

    # Ensure inputs are numpy arrays
    neutrino_velocity = np.asarray(neutrino_velocity)
    halo_velocity = np.asarray(halo_velocity)
    cdm_velocity = np.asarray(cdm_velocity)

    # Calculate velocity differences for neutrino-halo, neutrino-CDM, and CDM-halo
    v_nu_h = neutrino_velocity - halo_velocity
    v_nu_cdm = neutrino_velocity - cdm_velocity
    v_cdm_h = cdm_velocity - halo_velocity

    # Calculate dot products for various combinations of velocities
    v_nu_dot_v_h = np.dot(neutrino_velocity, halo_velocity)
    v_nu_dot_v_cdm = np.dot(neutrino_velocity, cdm_velocity)
    v_nuh_dot_v_h = np.dot(v_nu_h, halo_velocity)
    v_nuh_dot_v_nu = np.dot(v_nu_h, neutrino_velocity)
    v_nucdm_dot_v_cdm = np.dot(v_nu_cdm, cdm_velocity)
    v_nucdm_dot_v_nu = np.dot(v_nu_cdm, neutrino_velocity)
    v_cdm_dot_v_h = np.dot(cdm_velocity, halo_velocity)
    v_cdmh_dot_v_h = np.dot(v_cdm_h, halo_velocity)
    v_cdmh_dot_v_cdm = np.dot(v_cdm_h, cdm_velocity)

    # Calculate denominators as the dot product of each velocity vector with itself
    v_nu_dot_v_nu = np.dot(neutrino_velocity, neutrino_velocity)
    v_h_dot_v_h = np.dot(halo_velocity, halo_velocity)
    v_cdm_dot_v_cdm = np.dot(cdm_velocity, cdm_velocity)
        # "v_nuh · v_nuh",
        # "v_cdmh · v_cdmh",
        # "v_nucdm · v_nucdm"
    v_nuh_dot_v_nuh = np.dot(v_nu_h, v_nu_h)
    v_cdmh_dot_v_cdmh = np.dot(v_cdm_h, v_cdm_h)
    v_nucdm_dot_v_nucdm = np.dot(v_nu_cdm, v_nu_cdm)
    # Return the results as a flat array of scalar values
    return np.array([
        v_nu_dot_v_h, v_nu_dot_v_cdm, v_nuh_dot_v_h, v_nuh_dot_v_nu,
        v_nucdm_dot_v_cdm, v_nucdm_dot_v_nu, v_cdm_dot_v_h,
        v_cdmh_dot_v_h, v_cdmh_dot_v_cdm, v_nu_dot_v_nu,
        v_h_dot_v_h, v_cdm_dot_v_cdm, v_nuh_dot_v_nuh, v_cdmh_dot_v_cdmh, v_nucdm_dot_v_nucdm
    ])
import numpy as np
def prepare_metadata(spec, sim, Mass_cut, directory, boxsize, ngrid_list, n_h_threshold): #(spec, sim, Mass_cut, save_path, boxsize, ngrid_list, n_h_threshold)
    """
    Prepare metadata for the analysis.

    Parameters:
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - directory (str): Directory path of the data.
    - boxsize (float): Box size.
    - ngrid (int): Grid size.
    - n_h_threshold (int): Minimum number of halos in a sub-box for analysis.

    Returns:
    - metadata (dict): Metadata dictionary containing simulation information.
    """

    # Descriptions for the analysis
    x_description_list = [
        "v_nu · v_h",
        "v_nu · v_cdm",
        "v_nuh · v_h",
        "v_nuh · v_nu",
        "v_nucdm · v_cdm",
        "v_nucdm · v_nu",
        "v_cdm · v_h",
        "v_cdmh · v_h",
        "v_cdmh · v_cdm",
        "v_nu · v_nu",
        "v_h · v_h",
        "v_cdm · v_cdm",
        "v_nuh · v_nuh",
        "v_cdmh · v_cdmh",
        "v_nucdm · v_nucdm"
    ]

    metadata = {
        'specification': spec,
        'simulation': sim,
        'bulk_directory': directory,
        'mass_cut': f'{Mass_cut:.1e}',
        'boxsize': boxsize,
        'ngrid_list': ngrid_list,
        'n_h_threshold': n_h_threshold,
        'data_info': x_description_list
    }

    return metadata

def save_data(save_path, spec, sim, Mass_cut, n_h_threshold, metadata, data_store): ### Saving data
    """
    Save data.

    Parameters:
    - save_path (str): Path to save the data.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - metadata (dict): Metadata.
    - data_store (dict): Data to save.
    """
    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    directory = save_path+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    data_to_save = {
        'metadata': metadata,
        'data': data_store
    }
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}_nh_{n_h_threshold}.pickle'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'wb') as handle:
        pickle.dump(data_to_save, handle)


def analyze_and_save_for_ngrid(spec, sim, Mass_cut, ngrid_list, n_h_threshold, save_path, boxsize):
    """
    Analyzes data for each ngrid value, computes necessary quantities, and updates the data_store.

    Parameters:
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - ngrid_list (list): List of grid sizes to loop over.
    - n_h_threshold (int): the minimum number of halos in a sub-box to consider it
    - save_path (str): Directory to save the processed data.
    - boxsize: boxsize

    Returns:

    """
    file_path = f"/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/Runs_bulk_all_sims/{spec}/{sim}"
    metadata = prepare_metadata(spec, sim, Mass_cut, file_path, boxsize, ngrid_list, n_h_threshold);
    data_store ={}
    for ngrid in ngrid_list:
        # Call the function to load data
        cdm_bulk_all, nu_bulk_all, halo_bulk_all = load_data_bulk(file_path, ngrid, sim, spec, Mass_cut)

        num_sub_box = 0
        sub_box_data = halo_bulk_all['sub_box_data']
        results = np.zeros(15) # previously was 12
        for sub_box_key, sub_box_info in sub_box_data.items():

            if sub_box_key[0] < ngrid and sub_box_key[1] < ngrid and sub_box_key[2] < ngrid:
                
    
                # Calculate neutrino velocity based on the simulation condition
                if sim != "0.0ev":
                    neutrino_count = nu_bulk_all['sub_box_data'][sub_box_key]['N']
                    neutrino_velocity = nu_bulk_all['sub_box_data'][sub_box_key]['sum_bulk_vel'] / neutrino_count
                else:
                    neutrino_velocity = np.zeros_like(halo_bulk_all['sub_box_data'][sub_box_key]['sum_bulk_vel'])
    
                # Calculate halo and CDM velocities
                halo_count = halo_bulk_all['sub_box_data'][sub_box_key]['N']
                cdm_count = cdm_bulk_all['sub_box_data'][sub_box_key]['N']
    
                halo_velocity = halo_bulk_all['sub_box_data'][sub_box_key]['sum_bulk_vel'] / halo_count
                cdm_velocity = cdm_bulk_all['sub_box_data'][sub_box_key]['sum_bulk_vel'] / cdm_count
    
                # Accumulate results
                if (halo_count >= n_h_threshold):
                    num_sub_box += 1
                    results += compute_dot_products(neutrino_velocity, halo_velocity, cdm_velocity)

        # After looping over all sub-boxes, store results for the current ngrid
        if ngrid not in data_store:
            data_store[ngrid] = {}
        #         "v_nuh · v_nuh",
        # "v_cdmh · v_cdmh",
        # "v_nucdm · v_nucdm"
        data_store[ngrid] = {
            'v_nu·v_h': results[0],
            'v_nu·v_cdm': results[1],
            'v_nuh·v_h': results[2],
            'v_nuh·v_nu': results[3],
            'v_nucdm·v_cdm': results[4],
            'v_nucdm·v_nu': results[5],
            'v_cdm·v_h': results[6],
            'v_cdmh·v_h': results[7],
            'v_cdmh·v_cdm': results[8],
            'v_nu·v_nu': results[9],
            'v_h·v_h': results[10],
            'v_cdm·v_cdm': results[11],
            'v_nuh·v_nuh': results[12],
            'v_cdmh·v_cdmh': results[13],
            'v_nucdm·v_nucdm': results[14],
            'num_sub_box_analyzed': num_sub_box
        }

    save_data(save_path, spec, sim, Mass_cut, n_h_threshold, metadata, data_store)
    data_store.clear()
