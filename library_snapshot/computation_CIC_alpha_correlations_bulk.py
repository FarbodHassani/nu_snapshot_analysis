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


def perform_analysis_alpha_correlation(sims, specs, Mass_cuts, boxsize, save_path, ngrid_list, weight="rho1rho2", num_cores=1):
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

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    try:
        boxsize = np.float64(boxsize)

        # Let rank 0 compute and distribute work
        if rank == 0:
            total_combinations = len(Mass_cuts) * len(specs) * len(sims)
            num_cores = min(num_cores, total_combinations, size)

            if num_cores < 1:
                print("Error: No tasks to run.", file=sys.stderr)
                comm.Abort(1)

            if size > num_cores:
                print(f"[Rank 0] Warning: {size} ranks provided, but only {num_cores} will be used.", file=sys.stderr)

            combinations_per_process = total_combinations // num_cores
            remainder = total_combinations % num_cores

            for process_index in range(1, size):
                if process_index < num_cores:
                    start_index = process_index * combinations_per_process + min(process_index, remainder)
                    end_index = (process_index + 1) * combinations_per_process + min(process_index + 1, remainder)
                    comm.send((start_index, end_index), dest=process_index)
                else:
                    comm.send((None, None), dest=process_index)

            # Rank 0's work
            start_index = 0
            end_index = combinations_per_process + (1 if remainder > 0 else 0)

        else:
            try:
                start_index, end_index = comm.recv(source=0)
                if start_index is None:
                    comm.Barrier()
                    MPI.Finalize()
                    sys.exit(0)  # Exit unused ranks
            except Exception as e:
                print(f"[Rank {rank}] Failed to receive work assignment: {e}", file=sys.stderr)
                comm.Barrier()
                sys.exit(0)  # Exit gracefully on receive error

        # Process assigned combinations
        for index in range(start_index, end_index):
            try:
                Mass_cut_index = index % len(Mass_cuts)
                spec_index = (index // len(Mass_cuts)) % len(specs)
                sim_index = index // (len(Mass_cuts) * len(specs))

                Mass_cut = Mass_cuts[Mass_cut_index]
                spec = specs[spec_index]
                sim = sims[sim_index]

                analyze_and_save_for_ngrid(spec, sim, Mass_cut, ngrid_list, save_path, boxsize, weight)
            except Exception as e:
                print(f"[Rank {rank}] Skipping task (sim={sim}, spec={spec}, Mass_cut={Mass_cut}): {e}", file=sys.stderr)
                continue  # Skip to next task

        comm.Barrier()
        if rank == 0:
            print("Perform analysis finished!")

    except Exception as e:
        print(f"[Rank {rank}] Critical error: {e}", file=sys.stderr)
        comm.Abort(1)
    finally:
        pass


def load_data_bulk(file_path, ngrid, sim, spec, Mass_cut):
    """
    Load data for bulk velocities.
    Parameters:
    - file_path (str): Path to the data files.
    - ngrid (int): Number of grid points.
    - sim (str): Simulation identifier.
    - spec (str): Specification identifier.
    - Mass_cut (float): Mass cut value.
    Returns:
    - tuple: (cdm_bulk_all, nu_bulk_all, halo_bulk_all) if successful, None if any file is missing or an error occurs.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    try:
        cdm_analysis = True
        nu_analysis = True

        # Define file paths
        halo_all_path = f"{file_path}/{convert_mass_cuts([Mass_cut])[0]}/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_halo_mass_{Mass_cut:.1e}.pickle"
        cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_cdm.pickle"
        nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_nu.pickle" if sim != "0.0ev" else ""

        # Fallback paths for non-L_500_Ngrid_6144 cases
        if spec != "L_500_Ngrid_6144":
            if not os.path.exists(cdm_path):
                cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_cdm.pickle"
            if nu_path and not os.path.exists(nu_path):
                nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_ncdm0.pickle" if sim != "0.0ev" else ""

        # Check file existence
        required_paths = [cdm_path] if cdm_analysis else []
        if nu_analysis and nu_path:
            required_paths.append(nu_path)
        required_paths.append(halo_all_path)  # Halo data is always required

        for path in required_paths:
            if not os.path.exists(path):
                print(f"[Rank {rank}] Skipping task (sim={sim}, spec={spec}, Mass_cut={Mass_cut}, ngrid={ngrid}): File not found: {path}", file=sys.stderr)
                return None

        # Load data
        cdm_bulk_all = np.load(cdm_path, allow_pickle=True) if cdm_analysis else None
        nu_bulk_all = np.load(nu_path, allow_pickle=True) if nu_analysis and nu_path else None
        halo_bulk_all = np.load(halo_all_path, allow_pickle=True)

        return cdm_bulk_all, nu_bulk_all, halo_bulk_all

    except Exception as e:
        # print(f"[Rank {rank}] Skipping task (sim={sim}, spec={spec}, Mass_cut={Mass_cut}, ngrid={ngrid}): Error: {e}", file=sys.stderr)
        return None
    

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


def prepare_metadata(spec, sim, Mass_cut, directory, boxsize, ngrid_list, weight):
    """
    Prepare metadata for the analysis.

    Parameters:
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - directory (str): Directory path of the data.
    - boxsize (float): Size of the simulation box.
    - ngrid_list (list of int): List of grid sizes used for analysis.

    Returns:
    - metadata (dict): Dictionary containing metadata and data description.
    """

    metadata = {
        'specification': spec,
        'simulation': sim,
        'bulk_directory': directory,
        'mass_cut': f'{Mass_cut:.1e}',
        'boxsize': boxsize,
        'ngrid_list': ngrid_list,
        'weight method': weight,
        'data_info': (
            "Each 'x_y' entry in the data dictionary contains a dictionary with two sections:\n\n"
            "1. std_cal — the direct statistical calculation:\n"
            "   A dictionary of eight quantities:\n"
            "      • R        = ⟨v₁ · v₂⟩ / ⟨v₁²⟩          — normalized velocity correlation (projection of v₂ onto v₁).\n"
            "      • R_std    = √Var(R)                   — standard deviation of R, computed via error propagation.\n"
            "      • N_mean   = ⟨v₁ · v₂⟩                  — mean of the numerator (dot product of the two velocity fields).\n"
            "      • D_mean   = ⟨v₁ · v₁⟩                  — mean of the denominator (self-dot of the reference field).\n"
            "      • N_var    = Var(v₁ · v₂)              — variance of the numerator.\n"
            "      • D_var    = Var(v₁ · v₁)              — variance of the denominator.\n"
            "      • ND_cov   = Cov(v₁ · v₂, v₁ · v₁)     — covariance between numerator and denominator arrays.\n"
            "      • ND_mean  = ⟨(v₁ · v₂) × (v₁ · v₁)⟩   — mean of element-wise product of numerator and denominator arrays.\n\n"
            "2. Jackknife — resampling-based error estimation:\n"
            "   A dictionary of five quantities:\n"
            "      • R         = full-sample normalized velocity correlation.\n"
            "      • R_std_jk  = jackknife estimate of the standard deviation of R.\n"
            "      • K         = number of jackknife replicates.\n"
            "All averages and variances are computed over sub-boxes with valid densities. "
            "Jackknife partitions the grid into blocks and recomputes the statistic leaving one block out each time."
        )
    }

    return metadata


def save_data(save_path, spec, sim, Mass_cut, metadata, data_store): ### Saving data
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
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}.pickle'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'wb') as handle:
        pickle.dump(data_to_save, handle)


# def compute_correlations(bulk_all_1, bulk_all_2):
#     # Extract densities
#     rho_1 = bulk_all_1['sub_box_data']['density']
#     rho_2 = bulk_all_2['sub_box_data']['density']

#     # Define velocity components for both
#     p_x_1 = np.zeros_like(rho_1)
#     p_y_1 = np.zeros_like(rho_1)
#     p_z_1 = np.zeros_like(rho_1)

#     p_x_2 = np.zeros_like(rho_2)
#     p_y_2 = np.zeros_like(rho_2)
#     p_z_2 = np.zeros_like(rho_2)


#     weight_N = np.sum(rho_1 * rho_2, dtype=np.float64)
#     weight_D = np.sum(rho_1 * rho_1, dtype=np.float64) 

#     p_x_1 = bulk_all_1['sub_box_data']['P_x']
#     p_y_1 = bulk_all_1['sub_box_data']['P_y']
#     p_z_1 = bulk_all_1['sub_box_data']['P_z']

#     p_x_2 = bulk_all_2['sub_box_data']['P_x']
#     p_y_2 = bulk_all_2['sub_box_data']['P_y']
#     p_z_2 = bulk_all_2['sub_box_data']['P_z']

#     # Compute numerator and denominator arrays
#     N_array = (p_x_1 * p_x_2 + p_y_1 * p_y_2 + p_z_1 * p_z_2)/weight_N
#     D_array = (p_x_1**2 + p_y_1**2 + p_z_1**2)/weight_D


#     # Defensive checks
#     if D_array.size == 0 or np.mean(D_array) == 0:
#         print(f"Warning: D_mean zero or NaN for {Mass_cut} {spec} {sim} {ngrid}, skipping. Speculate!")
#         return [np.nan] * 8

#     N_mean = np.mean(N_array)
#     D_mean = np.mean(D_array)

#     if np.isnan(N_mean) or np.isnan(D_mean):
#         print(f"Warning: D_mean zero or NaN for {Mass_cut} {spec} {sim} {ngrid}, skipping. Speculate!")
#         return [np.nan] * 8

#     # Compute R and its uncertainty
#     R = N_mean / D_mean

#     N_var = np.mean(masked_N_array**2) - N_mean**2
#     D_var = np.mean(masked_D_array**2) - D_mean**2
#     ND_mean = np.mean(masked_N_array * masked_D_array)
#     ND_cov = ND_mean - N_mean * D_mean

#     R_var = (N_var / D_mean**2) + (N_mean**2 * D_var) / D_mean**4 - (2 * N_mean * ND_cov) / D_mean**3
#     R_var = max(R_var, 0)
#     R_std = np.sqrt(R_var)

#     return [R, R_std, N_mean, D_mean, N_var, D_var, ND_cov, ND_mean]


def compute_correlations(bulk_all_1, bulk_all_2, weight='rho1rho2',
                         min_blocks=2, max_blocks=6):
    """
    Compute velocity correlation ratio and uncertainties.
    Returns dict with 'std_cal' (direct calculation) and 'Jackknife' (resampling).
    Assumes bulk_all_*['sub_box_data'] is a structured ndarray with fields:
      'density', 'P_x', 'P_y', 'P_z', each (nx,ny,nz).
    """
    # --- Extract arrays ---
    sb1 = bulk_all_1['sub_box_data']
    sb2 = bulk_all_2['sub_box_data']
    rho_1 = sb1['density'].astype(np.float64)
    rho_2 = sb2['density'].astype(np.float64)

    p1x, p1y, p1z = sb1['P_x'].astype(np.float64), sb1['P_y'].astype(np.float64), sb1['P_z'].astype(np.float64)
    p2x, p2y, p2z = sb2['P_x'].astype(np.float64), sb2['P_y'].astype(np.float64), sb2['P_z'].astype(np.float64)

    # --- Base mask: only finiteness (no positivity yet) ---
    base_mask = (np.isfinite(rho_1) & np.isfinite(rho_2) &
                 np.isfinite(p1x) & np.isfinite(p1y) & np.isfinite(p1z) &
                 np.isfinite(p2x) & np.isfinite(p2y) & np.isfinite(p2z))

    # --- Per-cell raw dot products (full 3D) ---
    p1p2 = p1x*p2x + p1y*p2y + p1z*p2z
    p1p1 = p1x*p1x + p1y*p1y + p1z*p1z
    p2p2 = p2x*p2x + p2y*p2y + p2z*p2z
    # --- Build N_array, D_array as full 3D, with NaN where invalid ---
    # We avoid divide-by-zero warnings by using np.divide with a boolean `where`.
    if weight == 'rho1':
        # N = (P1·P2)/rho2  (requires rho2>0)
        N_array = np.full_like(p1p2, np.nan, dtype=np.float64)
        np.divide(p1p2, rho_2, out=N_array, where=(rho_2 > 0))

        # D = (P1·P1)/rho1  (requires rho1>0)
        D_array = np.full_like(p1p1, np.nan, dtype=np.float64)
        np.divide(p1p1, rho_1, out=D_array, where=(rho_1 > 0))

    elif weight == 'rho1rho2':
        # N = P1·P2         (no division)
        N_array = p1p2.astype(np.float64, copy=False)

        # D = (rho2/rho1)*(P1·P1)  (requires rho1>0)
        D_array = np.full_like(p1p1, np.nan, dtype=np.float64)
        np.divide(rho_2 * p1p1, rho_1, out=D_array, where=(rho_1 > 0))

    elif weight == 'rho2':
        # N = (P1·P2)/rho1            (requires rho1>0)
        N_array = np.full_like(p1p2, np.nan, dtype=np.float64)
        np.divide(p1p2, rho_1, out=N_array, where=(rho_1 > 0))

        # D = (rho2/rho1^2)*(P1·P1)   (requires rho1>0)
        D_array = np.full_like(p1p1, np.nan, dtype=np.float64)
        np.divide(rho_2 * p1p1, rho_1**2, out=D_array, where=(rho_1 > 0))

    elif weight == '1':
        N_array = p1p2 
        D_array_1 =p1p1
        D_array_2 =p2p2
        D_array = D_array_1

    else:
        raise ValueError("weight must be 'rho1', 'rho1rho2' or 'rho2'.")

    # Effective mask = base validity AND both contributions finite
    eff_mask = base_mask & np.isfinite(N_array) & np.isfinite(D_array)
    if not np.any(eff_mask):
        return {'std_cal': {}, 'Jackknife': {}}

    # Helper that applies an extra mask (for jackknife) on top of eff_mask
    def R_with_mask(extra_mask):
        m = eff_mask & extra_mask
        if not np.any(m):
            return np.nan, (np.nan, np.nan, None, None)
        Nm = np.mean(N_array[m])
        Dm =np.sqrt( np.mean(D_array_1[m]) * np.mean(D_array_2[m]))
        if Dm == 0 or not np.isfinite(Dm) or not np.isfinite(Nm):
            return np.nan, (np.nan, np.nan, None, None)
        return Nm / Dm, (Nm, Dm, N_array[m], D_array[m])
    
    # --- Standard calculation on full sample ---
    full_mask = np.ones_like(eff_mask, dtype=bool)
    R, (N_mean, D_mean, N_used, D_used) = R_with_mask(full_mask)

    if N_used is None or D_used is None:
        std_cal = {k: np.nan for k in ['R','R_std','N_mean','D_mean','N_var','D_var','ND_cov','ND_mean']}
    else:
        n = N_used.size
        if n > 1:
            N_var_cell = np.var(N_used, ddof=1)
            D_var_cell = np.var(D_used, ddof=1)
            ND_mean = np.mean(N_used * D_used)
            ND_cov_cell = ND_mean - N_mean * D_mean
            R_var = (N_var_cell/n)/(D_mean**2) \
                  + (N_mean**2 * D_var_cell/n)/(D_mean**4) \
                  - (2*N_mean*ND_cov_cell/n)/(D_mean**3)
            R_std = np.sqrt(max(R_var, 0.0))
        else:
            N_var_cell = D_var_cell = ND_cov_cell = ND_mean = np.nan
            R_std = np.nan

        std_cal = {
            'R': R, 'R_std': R_std,
            'N_mean': N_mean, 'D_mean': D_mean,
            'N_var': N_var_cell, 'D_var': D_var_cell,
            'ND_cov': ND_cov_cell, 'ND_mean': ND_mean
        }

    # --- Jackknife over blocks (auto factorization along each axis) ---
    nx, ny, nz = rho_1.shape
    def choose_blocks(n, bmin, bmax):
        for b in range(bmax, bmin-1, -1):
            if n % b == 0:
                return b
        return bmin  # fallback if no divisor in range

    bx, by, bz = (choose_blocks(nx, min_blocks, max_blocks),
                  choose_blocks(ny, min_blocks, max_blocks),
                  choose_blocks(nz, min_blocks, max_blocks))

    xs = np.array_split(np.arange(nx), bx)
    ys = np.array_split(np.arange(ny), by)
    zs = np.array_split(np.arange(nz), bz)

    R_full = R
    jk_vals = []
    for xx in xs:
        for yy in ys:
            for zz in zs:
                keep = np.ones((nx, ny, nz), dtype=bool)
                keep[np.ix_(xx, yy, zz)] = False  # drop this block
                R_lo, _ = R_with_mask(keep)
                if np.isfinite(R_lo):
                    jk_vals.append(R_lo)

    jk_vals = np.asarray(jk_vals, dtype=np.float64)
    K = jk_vals.size
    if K > 0:
        R_bar = np.mean(jk_vals)
        var_j = (K - 1)/K * np.sum((jk_vals - R_bar)**2)
        R_std_jk = np.sqrt(max(var_j, 0.0))
    else:
        R_bar = R_std_jk = np.nan

    Jackknife = {
        'R': R_full,
        'R_std_jk': R_std_jk,
        'K': K,
        'R_jk_mean': R_bar,
    }

    return {'std_cal': std_cal, 'Jackknife': Jackknife}


def analyze_and_save_for_ngrid(spec, sim, Mass_cut, ngrid_list, save_path, boxsize, weight):
    """
    Analyzes data for each ngrid value, computes necessary quantities, and updates the data_store.

    Parameters:
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - ngrid_list (list): List of grid sizes to loop over.
    - save_path (str): Directory to save the processed data.
    - boxsize: boxsize

    Returns:

    """
    file_path = f"/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/Run_Bulk_velocities_CIC/{spec}/{sim}"
    metadata = prepare_metadata(spec, sim, Mass_cut, file_path, boxsize, ngrid_list, weight);
    data_store ={}
    data_tmp={}
    for ngrid in ngrid_list:
        # Load data
        cdm_bulk_all, nu_bulk_all, halo_bulk_all = load_data_bulk(file_path, ngrid, sim, spec, Mass_cut) 
    
        # Initialize the per-ngrid data dictionary
        data_tmp = {
            'cdm_x_h': compute_correlations(cdm_bulk_all, halo_bulk_all, weight),
            'h_x_cdm': compute_correlations(halo_bulk_all, cdm_bulk_all, weight),
        }
    
        if sim != "0.0ev":
            data_tmp.update({
                'cdm_x_nu': compute_correlations(cdm_bulk_all, nu_bulk_all, weight),
                'nu_x_cdm': compute_correlations(nu_bulk_all, cdm_bulk_all, weight),
                'nu_x_h': compute_correlations(nu_bulk_all, halo_bulk_all, weight),
                'h_x_nu': compute_correlations(halo_bulk_all, nu_bulk_all, weight),
            })
    
        data_store[ngrid] = data_tmp

    save_data(save_path, spec, sim, Mass_cut, metadata, data_store)
    data_store.clear()




