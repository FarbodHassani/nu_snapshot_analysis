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

def compute_regression(Mass_cuts, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num=1,  n_h_threshold=3, run_tests=False, remove_extra_files=False, print_all_mass_cuts= False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Get the total number of processes
    size = comm.Get_size()

    if size > 1:
        # If more than 1 process, release extra cores
        if rank == 0:
            print("Releasing extra cores and running compute_regression with one core...")

        # Barrier to ensure all processes see the message
        comm.Barrier()

        # Split the communicator to isolate one process
        new_comm = comm.Split(color=(rank == 0), key=0)

        if rank == 0:
            # Run compute_regression_func only on the process with rank 0
            print("Computing with n_h_threshold = "+ str(n_h_threshold))
            compute_regression_func(Mass_cuts, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, n_h_threshold, run_tests, remove_extra_files, print_all_mass_cuts)

        # Barrier to ensure all processes finish before continuing
        comm.Barrier()

        # Free the new communicator
        new_comm.Free()

        if rank == 0:
            print("compute_regression finished successfully.")
    else:
        # If only 1 process, simply call compute_regression_func
        compute_regression_func(Mass_cuts, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, n_h_threshold, run_tests, remove_extra_files, print_all_mass_cuts)
        print("compute_regression finished successfully.")


def compute_regression_func(Mass_cuts, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, n_h_threshold,  run_tests, remove_extra_files, print_all_mass_cuts):
    """
    Final analysis function.
    Parameters:
    - Mass_cuts (array of float): Mass cut values.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - boxsize (float): Box size of the simulation.
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
    default_metadata = {
        'simulation': sim,
        'spec': spec,
        'boxsize': boxsize,
        'ngrid_max': ngrid_max,
        'halo_dir': halo_dir,
        'n_h_threshold': n_h_threshold
    }
    for Mass_cut in Mass_cuts:
        if print_all_mass_cuts == False:
            data_store = nested_dict(4, list)
        # progress_bar = tqdm(total=len(ngrid_list), desc="Progress", position=0, leave=True)
        for ngrid in ngrid_list:
            data_set = load_xy_data_ngrid(save_path, spec, sim, Mass_cut, ngrid)
            if 'data' in data_set and data_set['data'] != {}:
                for it in range(iteration_num):
                    if it==0:
                        computed_values = compute_values(data_set, 0., n_h_threshold) # For the first iteration we consider var_systematic = 0
                        data_store['mass='+f'{Mass_cut:.1e}']['ngrid='+str(ngrid)]['iteration='+str(it+1)] = {
                            'dx': boxsize / ngrid,
                            'Variance(beta)': computed_values['variance_beta_final'],
                            'beta[sum(w beta)/sum(w)]': computed_values['beta_final_way1'],
                            'beta[sum(xy)/sum(x^2)]': computed_values['beta_final_way2'],
                            'beta[sum((x-x_avg)(y-y_avg)/sum((x-x_avg)^2)]': computed_values['beta_final_way3'],
                            'alpha[sum(alpha)/n]': computed_values['mean_alpha_final'],
                            'Variance(alpha)': computed_values['var_alpha_final'],
                            'variance_sys': computed_values['variance_sys'],
                            'number_sub-boxes': computed_values['n_sub_box']
                        }
                    else:
                        variance_sys = computed_values['variance_sys']; # Previous variance!
                        computed_values = compute_values(data_set, variance_sys, n_h_threshold) # For the first iteration we consider var_systematic = 0
                        data_store['mass='+f'{Mass_cut:.1e}']['ngrid='+str(ngrid)]['iteration='+str(it+1)] = {
                            'dx': boxsize / ngrid,
                            'Variance(beta)': computed_values['variance_beta_final'],
                            'beta[sum(w beta)/sum(w)]': computed_values['beta_final_way1'],
                            'beta[sum(xy)/sum(x^2)]': computed_values['beta_final_way2'],
                            'beta[sum((x-x_avg)(y-y_avg)/sum((x-x_avg)^2)]': computed_values['beta_final_way3'],
                            'alpha[sum(alpha)/n]': computed_values['mean_alpha_final'],
                            'Variance(alpha)': computed_values['var_alpha_final'],
                            'variance_sys': computed_values['variance_sys'],
                            'number of sub-boxes': computed_values['n_sub_box']
                        }
            else:
                # Populate empty values if the data does not exist
                data_store['mass=' + f'{Mass_cut:.1e}']['ngrid=' + str(ngrid)] = {
                    'dx': boxsize / ngrid,
                    'Variance(beta)': None,
                    'beta[sum(w beta)/sum(w)]': None,
                    'beta[sum(xy)/sum(x^2)]': None,
                    'beta[sum(x-x_avg)(y-y_avg)/sum(x-x_avg)^2]': None,
                    'alpha[sum(alpha)/n]': None,
                    'Variance(alpha)': None,
                    'variance_sys': None,
                    'number_sub-boxes': None
                }
                metadata = default_metadata;

            # Update progress bar
            # progress_bar.update(1)
            if remove_extra_files:
                remove_files(save_path, spec, sim, Mass_cut, ngrid);
        # Close progress bar
        # progress_bar.close()

        if print_all_mass_cuts == False:
            if 'metadata' in data_set:
                metadata = data_set['metadata']
            else:
                metadata = default_metadata
            save_data_final(save_path, spec, sim, Mass_cut, metadata, data_store, print_all_mass_cuts, n_h_threshold)
            

    if print_all_mass_cuts == True:
        if 'metadata' in data_set:
            metadata = data_set['metadata'];
        else:
            metadata = default_metadata;
        save_data_final(save_path, spec, sim, Mass_cut, metadata, data_store, print_all_mass_cuts, n_h_threshold)

def load_xy_data_ngrid(save_path, spec, sim, Mass_cut, ngrid):
    """Loading data for each ngrid data.

    Parameters:
    - save_path (str): Path to save the data.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    """
    directory = os.path.join(save_path, f'{Mass_cut:.1e}')
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}_ngrid_{ngrid}.pickle'
    file_path = os.path.join(directory, file_name)

    try:
        with open(file_path, 'rb') as handle:
            loaded_data = pickle.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_name}' doesn't exist!")
        # print(f"Warning: The file '{file_name}' doesn't exist!")
        # loaded_data = {}  # Return an empty dictionary if the file is not found

    return loaded_data

def check_file_exist(save_path, spec, sim, Mass_cut, ngrid): ### Saving data
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
    try:
        with open(file_path, 'rb') as handle:
            loaded_data = pickle.load(handle)
    except FileNotFoundError:
        print("The file "+file_name+" doesn't exist!")
    return


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

def compute_values(data_set, variance_sys, n_h_threshold):
    """
    Compute various statistical values (regression) from the given data_set.

    Parameters:
    - data_set (dict): Dictionary containing sub-box data.

    Returns:
    - dict: Dictionary containing computed statistical values.
    """
    x_dot_y = [0.0]
    x_dot_x = [0.0]
    xy = [0.0]
    xx = [0.0]
    sum_x = [0.0]
    sum_y = [0.0]
    n_h_tot = 0 # total number of halos within all sub-boxes that used for the computation
    sum_w_ijk = [0.0]
    sum_beta_w_ijk = [0.0]
    mean_alpha_final = [0.0]
    mean_alpha_final_old = [0.0]
    var_alpha_final = [0.0]
    variance_beta_final = [0.0]
    beta_final_way1 = [0.0]
    beta_final_way2 = [0.0]
    beta_final_way3 = [0.0] # In this method we compute (Sum_over_halos x_i y_i - N y_bar x_bar)/(Sum_over_halos x_i x_i - N x_bar^2)
    ### Note that we already have save Sum x_i y_i for all halos within each sub-box that satisfy threshhol > n_h, and also we have computed sum y_i and sum x_i, so we can compute y_bar and x_bar
    #
    if n_h_threshold < 3:
        raise ValueError("n_h_threshold must be at least 3. Having fewer than 3 halos in a sub-box is insufficient for computing variance. Please increase n_h_threshold to 3 or higher.")
    ### First round would be to compute mu! I decided not to do it through online algorithm as it might become complicated to implement and derive the equation!
    for sub_box_index in data_set['data']:
        sub_box_data = data_set['data'][sub_box_index]
        if sub_box_data['n_h'] >= n_h_threshold:
            x_dot_y += sub_box_data['x_dot_y']
            x_dot_x += sub_box_data['x_dot_x']
            ###
            xy += sub_box_data['xy']
            xx += sub_box_data['xx']
            sum_x += sub_box_data['sum_x']
            sum_y += sub_box_data['sum_y']
            n_h_tot += sub_box_data['n_h'];

            w_ijk = 1. / (sub_box_data['variance_b'] + variance_sys) # variance_sys is the systematic part which is going to be find through iteration!
            sum_w_ijk += w_ijk # W_n = W_n-1 + w_ijk and W_n is Sum_i=1^n w_i
            sum_beta_w_ijk += (w_ijk * sub_box_data['b'])
            mean_alpha_final_old = mean_alpha_final;
            mean_alpha_final = mean_alpha_final + (w_ijk/sum_w_ijk) * (sub_box_data['alpha'] - mean_alpha_final)
            var_alpha_final = var_alpha_final + (w_ijk/sum_w_ijk) * ( (sub_box_data['alpha'] - mean_alpha_final) * (sub_box_data['alpha'] - mean_alpha_final_old) - var_alpha_final)


    if ( (not np.all( np.array(sum_w_ijk) == 0)) and (not np.all(np.array(x_dot_x) == 0)) ):

        if ( (np.any( np.array(sum_w_ijk) == 0)) and np.any(np.array(x_dot_x) == 0) ):
            print("Warning: Some elements are zero, division may not be possible.")
        else:
            variance_beta_final = 1.0 / sum_w_ijk
            beta_final_way1 = sum_beta_w_ijk / sum_w_ijk
            beta_final_way2 = x_dot_y / x_dot_x
            beta_final_way3 = (xy - (sum_x * sum_y)/n_h_tot) / (xx - (sum_x * sum_x)/n_h_tot) # Note that y_bar = sum_y/n_h_tot, x_bar = sum_x/n_h_tot -->  n_h_tot * x_bar * y_bar = sum_x * sum_y/n_h_tot

    ####### Round two of loops
    mu = beta_final_way1;
    sum_w_ijk_squared = [0.0];
    sum_term = [0.0];
    num_sub_boxes_analyzed = 0;
    for sub_box_index in data_set['data']:
        sub_box_data = data_set['data'][sub_box_index]
        if sub_box_data['n_h'] >= n_h_threshold:
            num_sub_boxes_analyzed += 1
            #############
            w_ijk = 1. / (sub_box_data['variance_b'] + variance_sys) # variance_sys is the systematic part which is going to be find through iteration!
            w_ijk_squared = w_ijk**2 # variance_sys is the systematic part which is going to be find through iteration!
            w_ijk_cubed = 1. / (sub_box_data['variance_b'] + variance_sys)**3 # variance_sys is the systematic part which is going to be find through iteration!
            sum_term += ((sub_box_data['b'] - mu)**2 - sub_box_data['variance_b']) * w_ijk_squared
            # sum_term_beta_w_squared += (sub_box_data['b'] - mu)**2 * w_ijk_squared;
            # sum_term_beta_w_cubed += (sub_box_data['b'] - mu)**2 * w_ijk_cubed;
            # sum_w_ijk += w_ijk # W_n = W_n-1 + w_ijk and W_n is Sum_i=1^n w_i
            sum_w_ijk_squared += w_ijk_squared # W_n = W_n-1 + w_ijk and W_n is Sum_i=1^n w_i

    # deltaX = (sum_w_ijk - sum_term_beta_w_squared)/(sum_w_ijk_squared - 2.0 * sum_term_beta_w_cubed) # delta sigma_s^2 = sigma_s^2(n+1) - sigma_s^2(n)
    # variance_sys = variance_sys + deltaX
    # variance_sys = sum_term/sum_w_ijk_squared;
    if (not np.all( np.array(sum_w_ijk_squared) == 0)):
        if (np.any( np.array(sum_w_ijk_squared) == 0)):
            print("Warning: Some elements are zero, division may not be possible.")
        else:
            variance_sys = sum_term/sum_w_ijk_squared;
            variance_sys[variance_sys < 0] = 0;

    return {
        'variance_beta_final': variance_beta_final,
        'beta_final_way1': beta_final_way1,
        'beta_final_way2': beta_final_way2,
        'beta_final_way3': beta_final_way3,
        'mean_alpha_final': mean_alpha_final,
        'var_alpha_final': var_alpha_final,
        'variance_sys': variance_sys, # systematic variance for beta variation!
        'n_sub_box' : num_sub_boxes_analyzed
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
    result = compute_values(sub_boxes, 0)
        
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


def save_data_final(save_path, spec, sim, Mass_cut, metadata, data_store, print_all_mass_cuts, n_h_threshold): ### Saving data
    """
    Save data.

    Parameters:
    - save_path (str): Path to save the data.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - metadata (dict): Metadata.
    - data_store (dict): Data to save.
    - If print_all_mass_cuts = True --> All the mass cuts are saved into a single file!
    - If print_all_mass_cuts = False --> each mass cut is saved separately in a single file!
    """

    # Update metadata with the new information
    metadata['n_h_threshold_postprocessing'] = n_h_threshold
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

    if print_all_mass_cuts == True:
        file_name = f'regression_data_{spec}_sim_{sim}.pickle'
        file_path = os.path.join(directory, file_name)
    
        with open(file_path, 'wb') as handle:
            dill.dump({'metadata': metadata, 'data': data_store}, handle)
    else:
        file_path = f"{directory}/regression_data_{spec}_sim_{sim}_mass_cut_{Mass_cut:.2e}.pickle"
    
        with open(file_path, 'wb') as handle:
            dill.dump({'metadata': metadata, 'data': data_store}, handle)


