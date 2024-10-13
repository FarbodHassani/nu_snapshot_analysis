import os
import time
import pickle
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')
from ReadParticles import *
from library_snapshot import readsnap

### Function to calculate Fermi Dirac distribution in the neutrino bulk velocity reference. For each N_grid we loop over massive neutrinos and subtract the bulk velocity it's located in and subtract its velocity from bulk and then compute the Fermi Dirac with respect to the new velocity!
def generate_histogram_data_nu_ref(m_nu, boxsize, n_grid_sim, n_grid_scale, file_path, bulk_velocity_path, file_index_z, save_directory, min_value=1e-6, max_value=1e1, num_bins=120, number_every=1, sim_type='gadget2', c = 3.0 * 10 ** 5, do_test = False):
    """
    Generate histogram data for given neutrino masses and save them.
    
    Parameters:
        m_nu (float): Neutrino mass in eV.
        file_path (str): Path to the directory containing neutrino files.
        save_directory (str): Directory where the histogram data will be saved.
        sim_type (str): Type of simulation ("JD" or "gadget2").
        c (float): Speed of light.
        min_value (float): Minimum value for log binning.
        max_value (float): Maximum value for log binning.
        num_bins (int): Number of bins for histogram.
        number_every (int): Number of samples per particle.
    Returns:
        None
    """
    start_time = time.time()  # Initialize the start time

    # Calculate log_bin_edges
    log_bin_edges = np.logspace(np.log10(min_value), np.log10(max_value), num=num_bins+1)

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    data = {'hist': np.zeros(num_bins), 'bin_edges': log_bin_edges.tolist()}

    # Progress bar
    with tqdm(total=_get_total_iterations(sim_type, file_path), desc='Processing') as pbar:
        data = process_gadget2_simulation(m_nu, boxsize, n_grid_sim, n_grid_scale, file_path, bulk_velocity_path, log_bin_edges, c, number_every, pbar, do_test)
    output_file = os.path.join(save_directory, f'histogram_data_Fermi_Dirac_L_{boxsize}_{n_grid_sim}_{sim_type}_mnu_{np.round(m_nu,3)}_snap_{file_index_z}_n_grid_{n_grid_scale}.pickle')
    save_histogram_data(data, output_file)

def process_gadget2_simulation(m_nu, boxsize, n_grid_sim, n_grid_scale, file_path, bulk_velocity_path, log_bin_edges, c, number_every, pbar, do_test):
    data_hist = {
        'hist': np.zeros(len(log_bin_edges) - 1),
        'hist_nu': np.zeros(len(log_bin_edges) - 1),
        'bin_edges': log_bin_edges.tolist()}
    head = readsnap.snapshot_header(file_path)
    num_files = head.filenum
    
    nu_bulk_all = load_data_nu_bulk(bulk_velocity_path);
    ptype = head.format
    for num_file in range(num_files):
        if n_grid_sim>1024:
            vel_data = readsnap.read_block(file_path + "." + str(num_file), "VEL ", ptype, True, 0, False, False,[0,n_grid_sim**3,0,0,0,0])  
            pos_data = readsnap.read_block(file_path + "." + str(num_file), "POS ", ptype, True, 0, False, False,[0,n_grid_sim**3,0,0,0,0])/1000.

        else:
            vel_data = readsnap.read_block(file_path + "." + str(num_file), "VEL ", ptype)
            pos_data = readsnap.read_block(file_path + "." + str(num_file), "POS ", ptype)/1000.

        head = readsnap.snapshot_header(file_path + "." + str(num_file))
        print("The file " + file_path + "." + str(num_file), "is loading, file number", str(num_file), ", number of pcl to be laoded: ", str(head.npart),
                  ", loaded num of particles:" + str(np.shape(vel_data)[0])) 

        ###########
        index_pos = np.int64((pos_data[:]/ boxsize)*n_grid_scale)
        ### Making a dictionary to avoid looping over halos for comparisons:
        # the pre-built map for quick lookup
        sub_box_halo_map ={}
        for i, halo_index in enumerate(index_pos):
            halo_tuple = tuple(halo_index)
            # print(halo_tuple)
            if halo_tuple not in sub_box_halo_map:
                sub_box_halo_map[halo_tuple] = []
            sub_box_halo_map[halo_tuple].append(i)
        ####
        filtered_dict = {}
        for sub_box_index, data in nu_bulk_all['sub_box_data'].items():
            # print(sub_box_index)
            if data['N']> 10: # Only the sub-boxes that have minimum n_h_threshold halos are kept! 
                 # note that n_h must be larger than 2, based on definition of variance and the fact that we need 3 points to compute in regression!
                filtered_dict[sub_box_index] = data
        
        sub_box_data = {}  # Dictionary to store sub-box data
        for sub_box_index, data in filtered_dict.items():
            if sub_box_index not in sub_box_halo_map:
                # print(f"Sub-box {sub_box_index} does not contain any halos.")
                continue  # Skip to the next sub-box if this one has no halos
        
            halo_indices_in_box = sub_box_halo_map[sub_box_index]
            n_h = len(halo_indices_in_box)
            condition_sub_box = np.array(halo_indices_in_box)
            vel_bulk = nu_bulk_all['sub_box_data'][sub_box_index]['sum_bulk_vel']/nu_bulk_all['sub_box_data'][sub_box_index]['N']            
            vel_data_sub_box = vel_data[halo_indices_in_box]
            v_norm_nu = np.sqrt( (vel_data_sub_box[::number_every, 0] - vel_bulk[0] )** 2 + (vel_data_sub_box[::number_every, 1] - vel_bulk[1] )** 2 + (vel_data_sub_box[::number_every, 2] - vel_bulk[2])** 2)
            v_norm = np.sqrt( (vel_data_sub_box[::number_every, 0] )** 2 + (vel_data_sub_box[::number_every, 1])** 2 +(vel_data_sub_box[::number_every, 2])** 2)
            # #####
            p = m_nu * v_norm / c
            Energies = p
            p_nu = m_nu * v_norm_nu / c
            Energies_nu = p_nu
            ######
            # Update histograms
            hist, _ = np.histogram(Energies, bins=log_bin_edges, density=False)
            data_hist['hist'] += hist  # 'hist' is now guaranteed to be initialized
        
            hist_nu, _ = np.histogram(Energies_nu, bins=log_bin_edges, density=False)
            data_hist['hist_nu'] += hist_nu  # 'hist_nu' is now guaranteed to be initialized
            pbar.update(1)  # Update progress bar
    return data_hist

def load_data_nu_bulk(file_path):  ### Loading the bulk velocities
    """
    Load data.
    Parameters:
    - file_path (str): Path to the data files.

    Returns:
    - nu_bulk_all: Loaded data.
    - Data based on analysis selection.

    """
    nu_path = f"{file_path}"

    if nu_path and not os.path.exists(file_path):
        print(f"Warning: {file_path} doesn't exist.")
            
    # Load the data if the files exist
    nu_bulk_all = np.load(file_path, allow_pickle=True)
    return nu_bulk_all


def _get_total_iterations(sim_type, file_path):
    if sim_type == "JD":
        return 512
    elif sim_type == "gadget2":
        head = readsnap.snapshot_header(file_path)
        return head.filenum

def save_histogram_data(data, output_file):
    """
    Save histogram data along with metadata to a file.

    Parameters:
        data (dict): Dictionary containing histogram data.
        output_file (str): Path to save the output file.

    Returns:
        None
    """
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data}, f)


def fermi_dirac_compute(p, mu, T):
    # T = 1.95*(1+z);
    k = 8.617333262145e-5  # Boltzmann constant in eV/K
    return (1.)/ (np.exp((p - mu) / (k * T)) + 1.) # E in our case is pc (c=1) = m_nu v_nu * c/c^2 = m_nu*v_nu/c [in eV]
