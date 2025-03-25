import numpy as np
import pandas as pd
import pickle
from mpi4py import MPI
import os
import sys
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/library_snapshot')
from ReadHalos import *
from ReadHalos_dm import *
from ReadParticles import *
import ast
import time
import math
from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from library_snapshot import computation_xy

#### Here we compute  $\frac{\langle(\vec v_h - \vec v_{bulk}).(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{bulk} \rangle}{\langle x.x\rangle}, x =(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{bulk} $ where $\vec v_{bulk}$ is the bulk velocity which can be computed from different quantities $v_{nu}$, $v_h$ with a certain mass cut and etc and $\vec v_h$ is the halo velocity and $x^i$ contains halo mass, bias and its velocity information and then we save the files in a directory

#### The point is study the mass dependence and also understand whye it is non-zero for $\Lambda$CDM case. So we are going to compute v_bulk
#### Based on Eq 30 of arXiv:1611.04589v2, the quantity that we really have to compute is  $\frac{\langle(\vec v_h - \vec v_{bulk}).(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{h \nu, 16 } \rangle}{\langle x.x\rangle}, x =(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{h \nu, 16 }$ where $v_{h \nu, 16 }$ is the relative difference between $v_{h \nu, 16 } = v_{\nu, 16 } - v_{h, 16 }$ in 16 Mpc/h and $v_h, v_{\nu}$ are both the bulk velocities. The point is that the non-zero variation of halos around their average velocity in each scale is due to the presence of massive neutrinos and the relative different between bulk velocity of massive neutrinos and halos are actually sourcing the non-zero variation of velocity of halos around their average velocity due to dynamical friction. This effect by definition should be 0 in $\Lambda$CDM by if instead of $v_{\nu h}$ we use another quantities like $v_{vh}$ might be non-zero and probably can be cumputed analytically. The point is the relation between $v_{vh}$ and  $v_{\nu h}$  is not always trivial, this can be measured through measuring correlation coefficent of different quantitie.

## TODO: Important! Note that in the original method of beta, we need to compute x_i - <x>, where <x> means the average over all halos, which is not considered properly here! Need to be improved!


#############
## Functions
#############

def perform_analysis(sims, specs, Mass_cuts, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, coeff_halo=1., num_cores=1, cdm_analysis=True, nu_analysis=True, n_h_threshold = 3):
    """
    Main function to perform analysis for different combinations of simulation parameters.
    Parameters:
    - sims (list): List of simulation identifiers.
    - specs (list): List of specification identifiers.
    - Mass_cuts (list): List of mass cut values.
    - boxsize (float): Box size of the simulation.
    - ngrid_max (int): Maximum number of grid points.
    - halo_dir (str): Directory path of the halo data.
    - save_path (str): Path to save the analyzed data.
    - ngrid_step (int): Step size for the grid points.
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
        # print(f"Processor {rank} is handling: Mass_cut = {Mass_cut:.2e}, Spec = {spec}, Sim = {sim}")

        # Perform analysis for the current combination
        analyze_all_ngrids(Mass_cut, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, coeff_halo, cdm_analysis, nu_analysis, n_h_threshold)

    # Wait for all processes to complete
    comm.Barrier()
    if rank == 0:
        print("Perform analysis finished!")

def analyze_all_ngrids(Mass_cut, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, coeff_halo, cdm_analysis, nu_analysis, n_h_threshold):
    """
    Main analysis function which goes through a list of ngrids and print out the sub-boxes and the important information for each ngrid and save it as a file!

    Parameters:
    - Mass_cut (float): Mass cut value.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - boxsize (float): Box size of the simulation.
    - ngrid_max (int): Maximum number of grid points.
    - halo_dir (str): Directory path of the halo data.
    - save_path (str): Path to save the analyzed data.
    - ngrid_step (int): Step size for the grid points.
    - ngrid_list (list): List of grid points to analyze.
    - n_h_threshold (int): min amount of halos in a sub-box which are going to enter the analysis 
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    # Load halo data based on simulation size (JD or FH)
    if boxsize == 500.:
        pos_halos, vel_halos, masses, biases, ratio_number_halos = load_halo_data_JD(halo_dir, spec, sim, Mass_cut)
    else:
        pos_halos, vel_halos, masses, biases, ratio_number_halos = load_halo_data(halo_dir, spec, sim, Mass_cut)

    # Print analysis information
    if rank==0:
        print(f"The analysis is being done for Mass_cut: {Mass_cut:.2e}")
    if ((sim == "0.0ev") and  (nu_analysis == True) and (rank == 0)):
        print("\033[38;5;214m warning: The neutrino analysis is requested while it's a non-neutrino run -- Automatically nu_analysis is set to False \033[0m")
        nu_analysis = False;
    # Analyze data for each ngrid
    for ngrid in ngrid_list:
        # Prepare metadata
        metadata = prepare_metadata(spec, sim, Mass_cut, ratio_number_halos, halo_dir, boxsize, ngrid, coeff_halo, cdm_analysis, nu_analysis, n_h_threshold)
        analyze_ngrid(ngrid,  save_path, metadata, spec, sim, boxsize, Mass_cut, pos_halos, vel_halos, masses, biases, coeff_halo, cdm_analysis, nu_analysis, n_h_threshold)

def analyze_ngrid(ngrid, save_path, metadata, spec, sim, boxsize, Mass_cut, pos_halos, vel_halos, masses, biases, coeff_halo, cdm_analysis, nu_analysis, n_h_threshold):
    """
    Analyze data for a specific ngrid.
    Parameters:
    - ngrid (int): Number of grids.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - pos_halos (array): Positions of halos.
    - vel_halos (array): Velocities of halos.
    - masses (array): Masses of halos.
    - biases (array): Biases of halos.
    - cdm_analysis (bool): Whether to add bulk velocities of cdm particles.
    - nu_analysis (bool): Whether to add bulk velocities of nu particles.
    - n_h_threshold (int): min amount of halos in a sub-box which are going to enter the analysis 

    """
    file_path = f"/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/Runs_bulk_all_sims/{spec}/{sim}"
    data_store = {}  # Dictionary to store ngrid_data data

    # Call the function
    result = load_data_bulk(file_path, ngrid, sim, spec, Mass_cut, cdm_analysis, nu_analysis)
    
    # Assign variables based on the values of cdm_analysis and nu_analysis
    cdm_bulk_all = None;
    nu_bulk_all = None;
    if cdm_analysis and nu_analysis:
        cdm_bulk_all, nu_bulk_all, halo_bulk_all = result
    elif cdm_analysis and not nu_analysis:
        cdm_bulk_all, halo_bulk_all = result
    elif nu_analysis and not cdm_analysis:
        nu_bulk_all, halo_bulk_all = result
    else:
        halo_bulk_all = result
        
    index_pos = np.int32( (pos_halos/boxsize) * ngrid) #
    # the pre-built map for quick lookup
    sub_box_halo_map ={}
    for i, halo_index in enumerate(index_pos):
        halo_tuple = tuple(halo_index)
        if halo_tuple not in sub_box_halo_map:
            sub_box_halo_map[halo_tuple] = []
        sub_box_halo_map[halo_tuple].append(i)
    ####
    filtered_dict = {}
    for sub_box_index, data in halo_bulk_all['sub_box_data'].items():
        # print(sub_box_index)
        if data['N']>= n_h_threshold: # Only the sub-boxes that have minimum n_h_threshold halos are kept! 
             # note that n_h must be larger than 2, based on definition of variance and the fact that we need 3 points to compute in regression!
            filtered_dict[sub_box_index] = data
    
    sub_box_data = {}  # Dictionary to store sub-box data
    for sub_box_index, data in filtered_dict.items():
        # halo_indices_in_box = sub_box_halo_map[sub_box_index]

        try:
            halo_indices_in_box = sub_box_halo_map[sub_box_index]
        except KeyError:
            print(f"KeyError for sub_box_index: {sub_box_index}. ngrid: {ngrid}, Mass_cut: {Mass_cut:.2e}, spec: {spec}, sim: {sim}")
            continue  # Optionally, you can skip this iteration if the key is missing
        n_h = len(halo_indices_in_box)
        condition_sub_box = np.array(halo_indices_in_box)
        
        x_dot_y, x_dot_x, xy, xx, sum_x, sum_y, b, alpha, Variance_beta, n_h, store_sub_box = analyze_sub_box(sub_box_index, condition_sub_box, ngrid, sim, spec, Mass_cut, cdm_bulk_all, nu_bulk_all, halo_bulk_all, pos_halos, vel_halos, masses, biases, coeff_halo, n_h, cdm_analysis, nu_analysis)
            
        # Store sub-box specific data
        if store_sub_box:
            sub_box_data[tuple(sub_box_index)] = {
                'x_dot_y': x_dot_y, # This is (x - <x>) (y - <y>) for all halos in a sub-box
                'x_dot_x': x_dot_x, # This is (x - <x>)^2 for all halos in a sub-box
                'xy': xy, # This is (\vec x . \vec y) for all halos in a sub-box 
                'xx': xx, # This is (\vec x . \vec x) for all halos in a sub-box
                'sum_x': sum_x, # This is \sum_i x_i for all halos in a sub-box, this is useful to compute final <x> which is the average of x overall halos in the simulations
                'sum_y': sum_y, # This is  \sum_i y_i for all halos in a sub-box, this is useful to compute final <x> which is the average of x overall halos in the simulations
                'b': b,
                'alpha': alpha,
                'variance_b': Variance_beta,
                'n_h': n_h,
            }
        else:
            print(f"The sub-box with sub_box_index: {sub_box_index}. ngrid: {ngrid}, Mass_cut: {Mass_cut:.2e}, spec: {spec}, sim: {sim} has been excluded from computation due to an issue!")
    save_data(save_path, spec, sim, Mass_cut, ngrid, metadata, sub_box_data)
    sub_box_data.clear()
    

def analyze_sub_box(sub_box_index, condition_sub_box, ngrid, sim, spec, Mass_cut, cdm_bulk_all, nu_bulk_all, halo_bulk_all, pos_halos, vel_halos, masses, biases, coeff_halo, n_h, cdm_analysis, nu_analysis):
    """
    Analyze data for a specific sub-box.

    Parameters:
    - sub_box_index (array): Index of the sub-box.
    - ngrid (int): Number of grids.
    - sim (str): Simulation identifier.
    - spec (str): Specification identifier.
    - Mass_cut (float): Mass cut value.
    - cdm_bulk_all (dict): CDM bulk data.
    - nu_bulk_all (dict): Neutrino bulk data.
    - halo_bulk_all (dict): Halo bulk data.
    - pos_halos (array): Positions of halos.
    - vel_halos (array): Velocities of halos.
    - masses (array): Masses of halos.
    - biases (array): Biases of halos.
    - n_h (int): number of halos.
    - cdm_analysis (bool): Whether to add bulk velocities of cdm particles.
    - nu_analysis (bool): Whether to add bulk velocities of nu particles.
    """
    # Extract data for the specified sub-box
    vel_halos_subBox, v_h_cell, mass, bias_h, f_h, f_h_avg, f_dot_v_h_average, v_c_cell, v_ch_cell, v_nu_cell, v_cnu_cell, v_nuh_cell, v_nuh2_i =  extract_sub_box_data(sim, sub_box_index, cdm_bulk_all, nu_bulk_all, halo_bulk_all, pos_halos, condition_sub_box, vel_halos, masses, biases, coeff_halo, cdm_analysis, nu_analysis)
        
    # Compute x_i and x_i_avg lists along with their descriptions
    x_i_list, x_i_avg_list = compute_x_lists(f_h, f_h_avg, f_dot_v_h_average, v_nuh_cell, v_nuh2_i, v_c_cell, v_h_cell, v_ch_cell, v_cnu_cell, coeff_halo, cdm_analysis, nu_analysis)

    # Compute regression parameters and variance
    x_dot_y, x_dot_x, xy, xx, sum_x, sum_y, b, alpha, Variance_beta, store_sub_box = compute_regression_params(x_i_list, x_i_avg_list, vel_halos_subBox, v_h_cell, n_h, sub_box_index, ngrid, sim, Mass_cut) 

    return x_dot_y, x_dot_x, xy, xx, sum_x, sum_y, b, alpha, Variance_beta, n_h, store_sub_box 

def extract_sub_box_data(sim, sub_box_index, cdm_bulk_all, nu_bulk_all, halo_bulk_all, pos_halos, condition_sub_box, vel_halos, masses, biases, coeff_halo, cdm_analysis, nu_analysis):
    """
    Extract data relevant to the specified sub-box.

    Returns:
    - vel_halos_subBox (array): Velocities of halos in the sub-box.
    - v_nu_cell (array): Neutrino bulk velocity in the sub-box.
    - v_h_cell (array): Average velocity of halos in the sub-box.
    - v_c_cell (array): Average velocity of CDM in the sub-box.
    - f_h_avg (array): Average of the f function for halos in the sub-box.
    - f_dot_v_h_average (array): Weighted average of halo velocities for the f function in the sub-box.
    - v_cnu_cell (array): Velocity difference between CDM and neutrinos in the sub-box.
    - v_nuh_cell (array): Velocity difference between neutrinos and halos in the sub-box.
    - v_nuh2_i (array): Velocity difference between halos and neutrinos for each halo in the sub-box.
    - v_ch_cell (array): Velocity difference between CDM and halos in the sub-box.
    - mass (array): Masses of halos in the sub-box.
    - bias_h (array): Biases of halos in the sub-box.
    - f_h (array): Coefficient computed for each halo in the sub-box.
    - n_h (int): Number of halos in the sub-box.
    - cdm_analysis (bool): Whether to add bulk velocities of cdm particles.
    - nu_analysis (bool): Whether to add bulk velocities of nu particles.
    """
    v_c_cell = np.zeros(3);
    v_ch_cell = np.zeros(3);
    v_nu_cell = np.zeros(3);
    v_nuh_cell = np.zeros(3);
    v_nuh2_i = np.zeros(3);
    v_cnu_cell = np.zeros(3);
    # Extract relevant data for the specified sub-box
    vel_halos_subBox = vel_halos[condition_sub_box]
    v_h_cell = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']
    f_h_avg = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_b_M']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']
    f_dot_v_h_average = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_b_M_vel_h']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']

    if cdm_analysis:
        v_c_cell = cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']
        v_ch_cell = v_c_cell - v_h_cell #v_c_cell - v_h_cell/2.

    if nu_analysis:
        if sim != "0.0ev":
            v_nu_cell = nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']
        else:
            v_nu_cell = np.array([0.,0.,0.])
        v_nuh_cell = v_nu_cell - v_h_cell/coeff_halo
        v_nuh2_i = v_nu_cell - vel_halos_subBox/coeff_halo
        
    if nu_analysis and cdm_analysis:        
        v_cnu_cell = v_c_cell - v_nu_cell
    
    # Additional data for regression
    mass = masses[condition_sub_box]
    bias_h = biases[condition_sub_box]
    f_h = (bias_h + (mass/(1.3*1.e14))**(0.85))

    return vel_halos_subBox, v_h_cell, mass, bias_h, f_h, f_h_avg, f_dot_v_h_average, v_c_cell, v_ch_cell, v_nu_cell, v_cnu_cell, v_nuh_cell, v_nuh2_i
        
def compute_regression_params(x_i_list, x_i_avg_list, vel_halos_subBox, v_h_cell, n_h, sub_box_index, ngrid, sim, Mass_cut):
    """
    Compute regression parameters.
# x_dot_y, x_dot_x, xy, xx, sum_x, sum_y, b, alpha, Variance_beta, n_h, store_sub_box 
    Parameters:
    - x_i_list (list): List of x_i arrays.
    - x_i_avg_list (list): List of average x_i arrays.
    - vel_halos_subBox (array): Velocities of halos in the sub-box.
    - v_h_cell (array): Average velocity of halos in the sub-box.
    - n_h (int): Number of halos.

    Returns:
    - x_dot_y (float): Dot product of x - <x> and y <y>.
    - x_dot_x (float): Dot product of x -<x> and x -<x>.
    - xy (float): Dot product of x and y (without subtracting averages)
    - xx (float): Dot product of x and x (without subtracting averages)
    - sum_x (float): summing over all x_i of all halos withing a sub-box
    - sum_y (float): summing over all y_i of all halos withing a sub-box
    - b (float): Slope of the regression line.
    - alpha (array): Intercept of the regression line.
    - Variance_beta (float): Variance of the slope.
    """
    y_i = vel_halos_subBox  # N*3 array; All velocities of halos within the sub-box
    y_avg = v_h_cell  # 1*3 array
    y_minus_yavg = y_i - y_avg  # N*3 array -- v^i - <v_h>; v^i = vel_h

    x_dot_y_list = []
    x_dot_x_list = []
    ######
    xy_list = []
    xx_list = []
    sum_x_list = []
    sum_y_list = []
    #####
    b_list = []
    alpha_list = []
    Variance_beta_list = []
    store_sub_box = True  # Initialize indicator as True

    for x_i, x_i_avg in zip(x_i_list, x_i_avg_list):
        x_minus_xavg = x_i - x_i_avg  # x-<x>; N*3 array
        x_dot_y = np.sum(np.sum(y_minus_yavg * x_minus_xavg, axis=1))  # (y-<y>).(x-<x>) this sum is the same as np.sum((x_minus_xavg * y_minus_yavg)[:,:])
        x_dot_x = np.sum(np.sum(x_minus_xavg * x_minus_xavg, axis=1))  # (x - <x>)(x-<x>)

        #### For new beta computation
        xy = np.sum(np.sum(y_minus_yavg * x_i, axis=1))  # (y.x this sum is the same as np.sum((x * y)[:,:])
        xx = np.sum(np.sum(x_i * x_i, axis=1))  # (x.x this sum is the same as np.sum((x * x)[:,:])
        sum_x = np.sum(np.sum(x_i, axis=1)) # sum of x_i
        sum_y = np.sum(np.sum(y_minus_yavg, axis=1)) # sum of y_i which is actually y_i - <y>|over sub-box    
        
        if (x_dot_x == 0.):
            print(f"WARNING: The coefficient in the regression cannot be calculated! Investigate what's going on! sub_box_index: {sub_box_index}, ngrid: {ngrid}, sim: {sim}, Mass_cut: {Mass_cut:.2e}, this sub-box is excluded!")
            store_sub_box = False
            
        b = x_dot_y / x_dot_x if x_dot_x != 0 else 0
          # Slope b_1 in the sub-box! y_i = b x_i + alpha
        alpha = np.mean(y_minus_yavg - b * x_i_avg)  # <y> - b <x> Note that here we take the mean of the whole y-y_bar and x_i_avg so we mix x,y,z components of velocities. However we could do 
        # alpha = np.mean(np.mean(np.transpose(y_minus_yavg - b * x_i_avg), axis=1)), but at the end it doesn't make any difference! Note that \alpha should be of order b*x_bar e.g., <v_h>
        x_dot_y_list.append(x_dot_y)
        x_dot_x_list.append(x_dot_x)
        xy_list.append(xy)
        xx_list.append(xx)
        sum_x_list.append(sum_x)
        sum_y_list.append(sum_y)
        b_list.append(b)
        alpha_list.append(alpha)

    x_dot_y = np.array(x_dot_y_list)
    x_dot_x = np.array(x_dot_x_list)
    ####
    xy = np.array(xy_list)
    xx = np.array(xx_list)    
    sum_x = np.array(sum_x_list)
    sum_y = np.array(sum_y_list)
    ####
    b = np.array(b_list)
    alpha = np.array(alpha_list)
        # Compute variance beta if store_sub_box is still True
    if store_sub_box:
        for i in range(np.shape(x_i_list)[0]):
            errors = y_minus_yavg - alpha[i] - b[i] * x_i_list[i]  # e_i = y_i - alpha - beta x_i; An N*3 array
            sum_errors2 = np.sum(errors * errors)  # sum of e_i^2
    
            # Check n_h and x_dot_x before dividing
            denominator = (n_h - 2) * (n_h - 1) * np.sum(x_dot_x[i])
            # TODO/FIXME -- This needs to be improved. We don't need to save those sub-boxes! Also check well!
            if denominator == 0:
                print(f"WARNING: Somethig is wrong As the denominator is 0! sub_box_index: {sub_box_index}, ngrid: {ngrid}, sim: {sim}, Mass_cut: {Mass_cut:.2e}");
                store_sub_box = False
                break  # Stop further processing if variance cannot be calculated
            else:
                Variance_beta = sum_errors2 / denominator  # Variance of the slope/ n_h-2 is from the formula!
            # Variance_beta = sum_errors2 / ((n_h-2) * (n_h-1) * np.sum(x_dot_x[i]))  # Variance of the slope/ n_h-2 is from the formula! 
                Variance_beta_list.append(Variance_beta)
                
    return x_dot_y, x_dot_x, xy, xx, sum_x, sum_y, b, alpha, np.array(Variance_beta_list), store_sub_box 


def compute_x_lists(f_h, f_h_avg, f_dot_v_h_average, v_nuh_cell, v_nuh2_i, v_c_cell, v_h_cell, v_ch_cell, v_cnu_cell, coeff_halo, cdm_analysis, nu_analysis):
    """
    Compute lists of x_i and x_i_avg.

    Parameters:
    - f_h (n_h*1 array): Coefficient computed for each halo in the sub-box.
    - f_h_avg (float): Average of the f function for halos in the sub-box.
    - f_dot_v_h_average (3*1 array): Average of the f_h * v_h  for halos in the sub-box.
    - v_nuh_cell (1*3 array): Velocity difference between neutrinos and halos in the sub-box.
    - v_nuh2_i (n_h*3 array): Velocity difference between halos and neutrinos for each halo in the sub-box.
    - v_c_cell (array): Average velocity of CDM in the sub-box.
    - v_h_cell (array): Average velocity of halos in the sub-box.
    - v_ch_cell (array): Velocity difference between CDM and halos in the sub-box.
    - v_cnu_cell (array): Velocity difference between CDM and neutrinos in the sub-box.
    - cdm_analysis (bool): Whether to add bulk velocities of cdm particles.
    - nu_analysis (bool): Whether to add bulk velocities of nu particles.
    
    Returns:
    - x_i_list (list): List of x_i arrays.
    - x_i_avg_list (list): List of x_i_avg arrays.
    """

    # note that x_i_list[0] == f_h * v_h_cell; which it self is a 3*N_halos vector
    #  x_i_list[1] == f_h * v_ch_cell; which it self is a N_halos*3 vector

    x_i_list = []
    x_i_avg_list = []
    f_h = f_h[:, np.newaxis]
    
    # halos analysis 
    x_i_list.append(f_h * v_h_cell) # element 0 of x_i_list which is a  N_halos*3 vector
    x_i_avg_list.append(f_h_avg * v_h_cell)
    
    if cdm_analysis:
        x_i_list.append(f_h * v_ch_cell) # element 1 of x_i_list which is a  N_halos*3 vector
        x_i_avg_list.append(f_h_avg * v_ch_cell)

        x_i_list.append(f_h * v_c_cell) # element 2 of x_i_list which is a  N_halos*3 vector
        x_i_avg_list.append(f_h_avg * v_c_cell)

    if nu_analysis:
        x_i_list.append(f_h * v_nuh_cell) # if neutrinos are requested! element 3 of x_i_list which is a  N_halos*3 vector
        x_i_avg_list.append(f_h_avg * v_nuh_cell)
        
        x_i_list.append(f_h * v_nuh2_i) # if neutrinos are requested! element 4 of x_i_list which is a  N_halos*3 vector
        x_i_avg_list.append(f_h_avg * v_nuh_cell - f_dot_v_h_average / coeff_halo)
        
    if nu_analysis and cdm_analysis:
        x_i_list.append(f_h * v_cnu_cell) # if both cdm and nu are requested: element 5 of x_i_list which is a  N_halos*3 vector
        x_i_avg_list.append(f_h_avg * v_cnu_cell)
    
    return x_i_list, x_i_avg_list # the shape of x_i_list is (6, N_halos, 3) -- so for each index we have N_halo * 3 which is basically f_h_avg * v_cell
    # The shape of x_i_avg_list (6, 3)

def x_lists_description(cdm_analysis, nu_analysis):
    """
    Compute lists of x_i and x_i_avg descriptions.

    Parameters:
    - cdm_analysis (bool): Whether to add bulk velocities of cdm particles.
    - nu_analysis (bool): Whether to add bulk velocities of nu particles.
    
    Returns:
    - x_description_list (list): List of descriptions for x_i arrays.
    """
    x_description_list = []
    x_description_list.append("x_i = f(b_i, M_i) <v_h>")
    
    if cdm_analysis:
        x_description_list.append("x_i = f(b_i, M_i) <v_cdm> - <v_h>/coeff")
        x_description_list.append("x_i = f(b_i, M_i) <v_cdm>")

    if nu_analysis:
        x_description_list.append("x_i = f(b_i, M_i) <v_nu> - <v_h>/coeff")
        x_description_list.append("x_i = f(b_i, M_i)(v_nu_cell - v_h^i/coeff)")
        
    if nu_analysis and cdm_analysis:
        x_description_list.append("x_i = f(b_i, M_i) <v_cdm> - <v_nu>")
    
    return x_description_list

def load(file):
    with open(file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

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

def load_data_bulk(file_path, ngrid, sim, spec, Mass_cut, cdm_analysis=False, nu_analysis=False):  ### Loading the bulk velocities
    """
    Load data.
    Parameters:
    - file_path (str): Path to the data files.
    - ngrid (int): Number of grid points.
    - sim (str): Simulation identifier.
    - spec (str): Specification identifier.
    - Mass_cut (float): Mass cut value.
    - cdm_analysis (bool): Whether to add bulk velocities of cdm particles.
    - nu_analysis (bool): Whether to add bulk velocities of nu particles.
    Returns:
    - cdm_bulk_all, nu_bulk_all, halo_bulk_all: Loaded data.
    - Data based on analysis selection.

    """
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
    if not cdm_analysis and not nu_analysis:
        return halo_bulk_all
        
    elif cdm_analysis and not nu_analysis:
        return cdm_bulk_all, halo_bulk_all
        
    elif not cdm_analysis and nu_analysis:
        return nu_bulk_all, halo_bulk_all
        
    elif cdm_analysis and nu_analysis:
        return cdm_bulk_all, nu_bulk_all, halo_bulk_all


def save_data(save_path, spec, sim, Mass_cut, ngrid, metadata, data_store): ### Saving data
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
    
    directory = save_path+"/"+f'{Mass_cut:.1e}'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    data_to_save = {
        'metadata': metadata,
        'data': data_store
    }
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}_ngrid_{ngrid}.pickle'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'wb') as handle:
        pickle.dump(data_to_save, handle)

def load_all_xy_data(save_path, spec, sim, Mass_cut, ngrid): ### Saving data
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
    
def load_halo_data_JD(halo_dir, spec, sim, Mass_cut, rank_total=512):
    """
    Load halo data using the JD method.

    Parameters:
    - rank_total (int): Total number of ranks.
    - halo_dir (str): Directory path of the halo data.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.

    Returns:
    - pos_halos (array): Positions of halos.
    - vel_halos (array): Velocities of halos.
    - masses (array): Masses of halos.
    - biases (array): Biases of halos.
    - ratio_number_halos (float): The ratio of the number of halos satisfying the mass cut 
      to the total number of halos.
      """
    pos_data=[];vel_data=[];mass_data=[]; number_tot=0;
    for rank_id in range(rank_total):
        ### In the case of halos!
        if sim=="0.0ev":
            input_file =  halo_dir+spec+"/"+sim+"/halos/0.000halo"+str(rank_id)+".dat"
            a = 1.;
            file_data = ReadHaloFile_lcdm(input_file, a)
            number_tot += np.shape(file_data[6])[0];
            mass_conditions = (file_data[6] >= Mass_cut)  
            file_data = np.array(file_data).T[mass_conditions]
            pos_data_add = file_data[:,:3]
            vel_data_add = file_data[:,3:6]
            mass_data_add =  file_data[:,6:7]
            pos_data.append(pos_data_add)
            vel_data.append(vel_data_add)
            mass_data.append(mass_data_add)         
        else:
            input_file =  halo_dir+spec+"/"+sim+"/halos/0.000halo"+str(rank_id)+".dat"
            a = 1.;
            file_data = ReadHaloFile_data(input_file, a)
            number_tot += np.shape(file_data[6])[0];
            mass_conditions = (file_data[6] >= Mass_cut)  # Assuming file_data[6] contains the values
            file_data = np.array(file_data).T[mass_conditions]
            pos_data_add = file_data[:,:3]
            vel_data_add = file_data[:,3:6]
            mass_data_add =  file_data[:,6:7]
            pos_data.append(pos_data_add)
            vel_data.append(vel_data_add)
            mass_data.append(mass_data_add) 

    pos_halos = np.vstack(pos_data)
    vel_halos = np.vstack(vel_data)
    masses = np.vstack(mass_data)
    if np.shape(masses)[0] == 0:
        print("ERROR: No halos found for the given mass cut")
        return None;
    biases = bias.haloBias(masses, model = 'sheth01', z = 0.0, mdef = '200m')
    ratio_number_halos = np.shape(masses)[0]/number_tot;
    return pos_halos, vel_halos, masses[:,0], biases[:,0], ratio_number_halos

def load_halo_data(halo_dir, spec, sim, Mass_cut):
    """
    Load halo data.

    Parameters:
    - halo_dir (str): Directory path of the halo data.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.

    Returns:
    - pos_halos (array): Positions of halos.
    - vel_halos (array): Velocities of halos.
    - masses (array): Masses of halos.
    - biases (array): Biases of halos.
    - ratio_number_halos (float): The ratio of the number of halos satisfying the mass cut 
      to the total number of halos.
    """
    halo_cat = np.loadtxt(halo_dir+spec+"/"+sim+"/output/halos/out_2.list") # Loading the full halo catalogue to loop over each halo
    mass_conditions = (halo_cat[:,20]>=Mass_cut)                
    halo_cat = halo_cat[mass_conditions]  # applying the mass condition
    pos_halos = halo_cat[:,8:11];
    vel_halos = halo_cat[:,11:14];
    masses = halo_cat[:,20];
    biases = bias.haloBias(masses, model = 'sheth01', z = 0.0, mdef = '200m')
    ratio_number_halos = np.shape(masses)[0]/np.shape(mass_conditions)[0];
    return pos_halos, vel_halos, masses, biases, ratio_number_halos


def prepare_metadata(spec, sim, Mass_cut, ratio_number_halos, halo_dir, boxsize, ngrid, coeff_halo, cdm_analysis, nu_analysis, n_h_threshold):
    """
    Prepare metadata for the analysis.

    Parameters:
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    - Mass_cut (float): Mass cut value.
    - ratio_number_halos (float): The ratio of the number of halos satisfying the mass cut 
      to the total number of halos.
    - halo_dir (str): Directory path of the halo data.
    - boxsize (float): Box size.
    - ngrid (int): Grid size.
    - n_h_threshold (int): min amount of halos in a sub-box which are going to enter the analysis 

    Returns:
    - metadata (dict): Metadata dictionary containing simulation information.
    """
    x_description_list = x_lists_description(cdm_analysis, nu_analysis)
    metadata = {
        'specification': spec,
        'simulation': sim,
        'mass_cut': f'{Mass_cut:.1e}',
        'ratio_number_halos': ratio_number_halos,
        'halo_directory': halo_dir,
        'boxsize': boxsize,
        'ngrid': ngrid,
        'coeff(v_nu - v_h/coeff)': coeff_halo,
        'adding bulk velocity of cdm:':cdm_analysis,
        'adding bulk velocity of neutrinos:':nu_analysis,
        'x_description_list': x_description_list,
        'n_h_threshold': n_h_threshold 
               }
    return metadata



