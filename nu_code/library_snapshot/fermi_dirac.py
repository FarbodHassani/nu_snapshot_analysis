import os
import time
import pickle
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')
from ReadParticles import *
from library_snapshot import readsnap


def generate_histogram_data(m_nu, boxsize, n_grid, file_path, file_index_z, save_directory, min_value=1e-5, max_value=1e1, num_bins=100, number_every=1, sim_type='gadget2', c = 3.0 * 10 ** 5, do_test = False):
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
        if sim_type == "JD":
            data = process_JD_simulation(m_nu, file_path, log_bin_edges, c, number_every, pbar, do_test)
        elif sim_type == "gadget2":
            data = process_gadget2_simulation(m_nu, file_path, n_grid, log_bin_edges, c, number_every, pbar, do_test)

    output_file = os.path.join(save_directory, f'histogram_data_Fermi_Dirac_L_{boxsize}_n_grid_{n_grid}_{sim_type}_mnu_{np.round(m_nu,3)}_snap_{file_index_z}.pickle')
    save_histogram_data(data, output_file)

def process_JD_simulation(m_nu, file_path, log_bin_edges, c, number_every, pbar, do_test):
    data = {'hist': np.zeros(len(log_bin_edges) - 1), 'bin_edges': log_bin_edges.tolist()}
    rank_max  = 512
    if do_test:
        print("The test being done, only 3 filed will be loaded")
        rank_max=3
    for rank in range(rank_max):
        input_file = f"{file_path}/0.000xv{rank}_nu.dat"
        file_data = ReadParticleFile(input_file)
        vel_data = np.vstack((file_data[3], file_data[4], file_data[5])).T
        # vel_data = file_data[3:6]
        v_norm = np.sqrt(vel_data[::number_every, 0] ** 2 + vel_data[::number_every,1] ** 2 + vel_data[::number_every,2] ** 2)
        p = m_nu * v_norm / c
        Energies = p
        hist, _ = np.histogram(Energies, bins=log_bin_edges, density=False)
        data['hist'] += hist
        pbar.update(1)  # Update progress bar
    return data

def process_gadget2_simulation(m_nu, file_path, n_grid, log_bin_edges, c, number_every, pbar, do_test):
    data = {'hist': np.zeros(len(log_bin_edges) - 1), 'bin_edges': log_bin_edges.tolist()}
    head = readsnap.snapshot_header(file_path)
    num_files = head.filenum
    if do_test:
        print("The test being done, only 2 filed will be loaded")
        num_files = 3
    ptype = head.format
    for num_file in range(num_files):
        if n_grid>1024:
            vel_data = readsnap.read_block(file_path, "VEL ", ptype, True, 0, False, False,[0,n_grid**3,0,0,0,0])  
        else:
            vel_data = readsnap.read_block(file_path, "VEL ", ptype)
        head = readsnap.snapshot_header(file_path + "." + str(num_file))
        print("The file " + file_path + "." + str(num_file), "is loading, file number", str(num_file), ", number of pcl to be laoded: ", str(head.npart),
                  ", loaded num of particles:" + str(np.shape(vel_data)[0]))        
        v_norm = np.sqrt(vel_data[::number_every, 0] ** 2 + vel_data[::number_every,1] ** 2 + vel_data[::number_every,2] ** 2)
        p = m_nu * v_norm / c
        Energies = p
        hist, _ = np.histogram(Energies, bins=log_bin_edges, density=False)
        data['hist'] += hist
        pbar.update(1)  # Update progress bar
    return data

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
