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


def nested_dict(n, type): # definingnested dictionary!
    if n==1:
        defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))


def compute_regression(Mass_cuts, spec, sim, L, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num=1, run_tests=False, remove_extra_files=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Releasing extra cores and running compute_regression with one core...")

    # Get the total number of processes
    size = comm.Get_size()

    if size == 1:
        # Perform the computation
        compute_regression_func(Mass_cuts, spec, sim, L, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, run_tests, remove_extra_files)
        return 

    # Spawn a new process
    else:
        new_comm = comm.Spawn(target=compute_regression_func, args=(Mass_cuts, spec, sim, L, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, run_tests, remove_extra_files))
    
        # Wait for the spawned process to finish
        new_comm.Barrier()

    if rank == 0:
        print("compute_regression finished successfully.")

def compute_regression_func(Mass_cuts, spec, sim, L, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, run_tests, remove_extra_files):
    """
    Final analysis function.
    Parameters:
    - Mass_cuts (array of float): Mass cut values.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - L (float): Box size of the simulation.
    - ngrid_max (int): Maximum number of grid points.
    - halo_dir (str): Directory path of the halo data.
    - save_path (str): Path to save the analyzed data.
    - ngrid_step (int): Step size for the grid points.
    - ngrid_list (list): List of grid points to analyze.
    - save_results (bool): Flag to indicate whether to save results.
    - iteration_num (int): number of iteration for the computation of the estimators. In each iteration we have a better guess for sigma_sysetamic.
    - run_tests (bool): Flag to indicate whether to run tests.
    - remove_extra_files (bool): Flag to remove each ngrid files after computation is done!
    Returns:
    - int if run_tests is True, indicating successful test completion.
    - dict if save_results is False, containing metadata and computed data.
    - None if save_results is True, indicating successful saving of results.
    """

    if run_tests:
        test_computation();
        return 0;
        
    # data_store = {}
    data_store = nested_dict(4, list)
    for Mass_cut in Mass_cuts:
        progress_bar = tqdm(total=len(ngrid_list), desc="Progress", position=0, leave=True)
        for ngrid in ngrid_list:
            data_set = load_xy_data_ngrid(save_path, spec, sim, Mass_cut, ngrid)
            for it in range(iteration_num):
                if it==0:
                    computed_values = compute_values(data_set, 0) # For the first iteration we consider var_systematic = 0
                    data_store['mass='+f'{Mass_cut:.1e}']['ngrid='+str(ngrid)]['iteration='+str(it+1)] = {
                        'dx': L / ngrid,
                        'Variance(beta)': computed_values['variance_beta_final'],
                        'beta[sum(w beta)/sum(w)]': computed_values['beta_final_way1'],
                        'beta[sum(xy)/sum(x^2)]': computed_values['beta_final_way2'],
                        'alpha[sum(alpha)/n]': computed_values['mean_alpha_final'],
                        'Variance(alpha)': computed_values['var_alpha_final']
                    }
                else:
                    variance_tot = computed_values['variance_beta_final']; # Previous variance!
                    computed_values = compute_values(data_set, variance_tot) # For the first iteration we consider var_systematic = 0
                    data_store['mass='+f'{Mass_cut:.1e}']['ngrid='+str(ngrid)]['iteration='+str(it+1)] = {
                        'dx': L / ngrid,
                        'Variance(beta)': computed_values['variance_beta_final'],
                        'beta[sum(w beta)/sum(w)]': computed_values['beta_final_way1'],
                        'beta[sum(xy)/sum(x^2)]': computed_values['beta_final_way2'],
                        'alpha[sum(alpha)/n]': computed_values['mean_alpha_final'],
                        'Variance(alpha)': computed_values['var_alpha_final']
                    }

            # Update progress bar
            progress_bar.update(1)
            if remove_extra_files:
                remove_files(save_path, spec, sim, Mass_cut, ngrid);
        # Close progress bar
        progress_bar.close()
    
    metadata = data_set['metadata']    
    save_data_final(save_path, spec, sim, Mass_cut, metadata, data_store)

def load_xy_data_ngrid(save_path, spec, sim, Mass_cut, ngrid): ### Saving data
    """
    loading data for each ngrid data.

    Parameters:
    - save_path (str): Path to save the data.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - data_store (dict): Data to save.
    """
    directory = save_path+"/"+f'{Mass_cut:.1e}'
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}_ngrid_{ngrid}.pickle'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'rb') as handle:
        loaded_data = pickle.load(handle)
    return loaded_data

import os

def remove_files(save_path, spec, sim, Mass_cut, ngrid):
    """
    Remove data file for a specific combination of parameters.

    Parameters:
    - save_path (str): Path to save the data.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - ngrid (int): Number of grid points.
    """
    directory = os.path.join(save_path, f'{Mass_cut:.1e}')
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}_ngrid_{ngrid}.pickle'
    file_path = os.path.join(directory, file_name)

    try:
        os.remove(file_path)
        print(f"File '{file_path}' removed successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the file '{file_path}': {str(e)}")




def compute_values(data_set, variance_tot):
    """
    Compute various statistical values (regression) from the given data_set.

    Parameters:
    - data_set (dict): Dictionary containing sub-box data.

    Returns:
    - dict: Dictionary containing computed statistical values.
    """

    x_dot_y = 0.
    x_dot_x = 0.
    sum_w_ijk = 0.
    sum_beta_w_ijk = 0.
    mean_alpha_final = 0.
    mean_alpha_final_old = 0.
    var_alpha_final = 0.
    variance_sys = variance_tot/(len(data_set['data'])) # we estimate the systematic variance by the sample variance, using the variance found in the previous iteration! variance_sys = variance_tot/number of sub-boxes
    for sub_box_index in data_set['data']:
        sub_box_data = data_set['data'][sub_box_index]
        x_dot_y += sub_box_data['x_dot_y']
        x_dot_x += sub_box_data['x_dot_x']
        w_ijk = 1. / (sub_box_data['variance_b'] + variance_sys) # variance_sys is the systematic part which is going to be find through iteration!
        sum_w_ijk += w_ijk # W_n = W_n-1 + w_ijk and W_n is Sum_i=1^n w_i
        sum_beta_w_ijk += (w_ijk * sub_box_data['b'])
        mean_alpha_final_old = mean_alpha_final;
        mean_alpha_final = mean_alpha_final + (w_ijk/sum_w_ijk) * (sub_box_data['alpha'] - mean_alpha_final)
        var_alpha_final = var_alpha_final + (w_ijk/sum_w_ijk) * ( (sub_box_data['alpha'] - mean_alpha_final) * (sub_box_data['alpha'] - mean_alpha_final_old) - var_alpha_final)

    variance_beta_final = 1. / sum_w_ijk
    beta_final_way1 = sum_beta_w_ijk / sum_w_ijk
    beta_final_way2 = x_dot_y / x_dot_x

    return {
        'variance_beta_final': variance_beta_final,
        'beta_final_way1': beta_final_way1,
        'beta_final_way2': beta_final_way2,
        'mean_alpha_final': mean_alpha_final,
        'var_alpha_final': var_alpha_final,
        'variance_sys': variance_sys # systematic variance for beta variation!
    }


def test_computation():
    # Define simple numbers for test arrays
    # Sample sub-box data containing arrays for x_dot_y, x_dot_x, b, alpha, and variance_b
    # Call the computation function to calculate results
    # Define expected results for Variance(beta), beta[sum(w beta)/sum(w)], beta[sum(xy)/sum(x^2)],
    # alpha[sum(alpha)/n], and Variance(alpha)
    # Compare computed results with expected results

    
    sub_boxes = {
        'data': {
            (0, 0, 0): {
                'x_dot_y': np.array([1, 1, 2]),
                'x_dot_x': np.array([1, 1, 2]),
                'b': np.array([2, 1, 2]),
                'alpha': np.array([2, 1, 5]),
                'variance_b': np.array([0.5, 0.25, 0.75]),
                'n_h': 4
            },
            (0, 0, 1): {
                'x_dot_y': np.array([3, 1, 3]),
                'x_dot_x': np.array([1, 2, 4]),
                'b': np.array([1, 2, 3]),
                'alpha': np.array([0, 1, 2]),
                'variance_b': np.array([0.75, 0.25, 0.5]),
                'n_h': 7
            },
            (0, 1, 0): {
                'x_dot_y': np.array([2, 1, 2]),
                'x_dot_x': np.array([1, 3, 1]),
                'b': np.array([1, 2, 1]),
                'alpha': np.array([1, 4, 2]),
                'variance_b': np.array([0.5, 0.25, 0.5]),
                'n_h': 9
            }
        }
    }
    
    print("Data being tested:")
    print(sub_boxes)
    # Call the computation function
    result = compute_values(sub_boxes)
        
    # Define expected results
    expected_variance_beta = np.array([0.19, 0.08, 0.19])
    expected_beta_final_way1 = np.array([1.38, 1.67, 2.0])
    expected_beta_final_way2 = np.array([2, 0.5, 1])
    expected_alpha_final_mean = np.array([1, 2, 3])
    expected_alpha_final_var = np.array([0.67, 2.0, 2.0])

    # Compare with expected results
    print("\nTesting Results:")
    print("=====================================")
    print("Expected Variance Beta Final:", expected_variance_beta)
    print("Result Variance Beta Final:", np.round(result['variance_beta_final'],2))
    print("Agreement:", np.allclose(np.round(result['variance_beta_final'],2), expected_variance_beta))
    
    print("Expected Rounded Beta Final Way1:", expected_beta_final_way1)
    print("Result Rounded Beta Final Way1:", np.round(result['beta_final_way1'], 2))
    print("Agreement:", np.allclose(np.round(result['beta_final_way1'], 2), expected_beta_final_way1))
    
    print("Expected Beta Final Way2:", expected_beta_final_way2)
    print("Result Beta Final Way2:", result['beta_final_way2'])
    print("Agreement:", np.allclose(result['beta_final_way2'], expected_beta_final_way2))
    
    print("Expected Mean Alpha Final:", expected_alpha_final_mean)
    print("Result Mean Alpha Final:", result['mean_alpha_final'])
    print("Agreement:", np.allclose(result['mean_alpha_final'], expected_alpha_final_mean))
    
    print("Expected Rounded Variance Alpha Final:", expected_alpha_final_var)
    print("Result Rounded Variance Alpha Final:", np.round(result['var_alpha_final'], 2))
    print("Agreement:", np.allclose(np.round(result['var_alpha_final'], 2), expected_alpha_final_var))

    # Count the number of failed tests
    failed_tests = sum([
        not np.allclose(np.round(result['variance_beta_final'],2), expected_variance_beta),
        not np.allclose(np.round(result['beta_final_way1'], 2), expected_beta_final_way1),
        not np.allclose(np.round(result['beta_final_way2'], 2), expected_beta_final_way2),
        not np.allclose(np.round(result['mean_alpha_final'],2), expected_alpha_final_mean),
        not np.allclose(np.round(result['var_alpha_final'], 2), expected_alpha_final_var)
    ])

    print("\nSummary:")
    print("=====================================")
    print("Number of failed tests:", failed_tests)


def save_data_final(save_path, spec, sim, Mass_cut, metadata, data_store): ### Saving data
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
    
    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    data_to_save = {
        'metadata': metadata,
        'data': data_store
    }
    file_name = f'regression_data_{spec}_sim_{sim}.pickle'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'wb') as handle:
        dill.dump({'metadata': metadata, 'data': data_store}, handle)


