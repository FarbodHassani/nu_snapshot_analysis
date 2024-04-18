### In this code we compute <xy>/<x><y> where x =(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{h \nu, 16 } more efficently! We use MPI to make it fast!

import numpy as np
import pandas as pd
import matplotlib as mpl
import pickle
from mpi4py import MPI
import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
Colors = sns.color_palette("colorblind", 16).as_hex()
import matplotlib.pyplot as plt
text_size=26
fig_size_x=24
fig_size_y=14
from scipy.interpolate import interp1d
import pandas as pd
import ast
import time
from collections import defaultdict
import math
from itertools import product

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))
from collections import defaultdict
from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');


#### Here we compute  $\frac{\langle(\vec v_h - \vec v_{bulk}).(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{bulk} \rangle}{\langle x.x\rangle}, x =(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{bulk} $ where $\vec v_{bulk}$ is the bulk velocity which can be computed from different quantities $v_{nu}$, $v_h$ with a certain mass cut and etc and $\vec v_h$ is the halo velocity and $x^i$ contains halo mass, bias and its velocity information and then we save the files in a directory

#### The point is study the mass dependence and also understand whye it is non-zero for $\Lambda$CDM case. So we are going to compute v_bulk
#### Based on Eq 30 of arXiv:1611.04589v2, the quantity that we really have to compute is  $\frac{\langle(\vec v_h - \vec v_{bulk}).(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{h \nu, 16 } \rangle}{\langle x.x\rangle}, x =(b_h + (\frac{M_h}{1.3 \times 10^{14}})^{0.85})\vec v_{h \nu, 16 }$ where $v_{h \nu, 16 }$ is the relative difference between $v_{h \nu, 16 } = v_{\nu, 16 } - v_{h, 16 }$ in 16 Mpc/h and $v_h, v_{\nu}$ are both the bulk velocities. The point is that the non-zero variation of halos around their average velocity in each scale is due to the presence of massive neutrinos and the relative different between bulk velocity of massive neutrinos and halos are actually sourcing the non-zero variation of velocity of halos around their average velocity due to dynamical friction. This effect by definition should be 0 in $\Lambda$CDM by if instead of $v_{\nu h}$ we use another quantities like $v_{vh}$ might be non-zero and probably can be cumputed analytically. The point is the relation between $v_{vh}$ and  $v_{\nu h}$  is not always trivial, this can be measured through measuring correlation coefficent of different quantitie.


#################
## Global variables
#################
sims= ["0.0ev", "0.15ev", "0.3ev" , "0.6ev"];
specs = ["L_2048_Ngrid_2048"]# specs = ["L_1024_Ngrid_512"]#, "L_1024_Ngrid_1024"]
Mass_cuts = [1.e12, 5.e12, 1.e13, 5.e13, 7.e13];
L = 2048.
ngrid_max = 180
halo_dir = "/mn/stornext/u3/hassanif/neutrino_niayesh/simulations/"
save_path = "./correlations_xy/"
ngrid_step = 1
ngrid_list = range(1, ngrid_max, ngrid_step)

#################
## MPI part
#################
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#############
## Functions
#############
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
    """
    cdm_path = f"{file_path}/cdm/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_cdm.pickle"
    nu_path = f"{file_path}/nu/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_snap002_ncdm0.pickle" if sim != "0.0ev" else ""
    halo_all_path = f"{file_path}/{convert_mass_cuts([Mass_cut])[0]}/output/data_ngrid_{ngrid}_sim_{sim}_{spec}_halos_out_2.pickle"
    
    # Check if files exist, raise an error if not
    for path in [cdm_path, nu_path, halo_all_path]:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"{path} doesn't exist.")
    # Load the data
    cdm_bulk_all = np.load(cdm_path, allow_pickle=True)
    nu_bulk_all = np.load(nu_path, allow_pickle=True) if sim != "0.0ev" else None
    halo_bulk_all = np.load(halo_all_path, allow_pickle=True)
    return cdm_bulk_all, nu_bulk_all, halo_bulk_all


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
    
    data_to_save = {
        'metadata': metadata,
        'data': data_store
    }
    file_name = f'data_correlations_{spec}_sim_{sim}_mass_{Mass_cut:.1e}.pickle'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'wb') as handle:
        pickle.dump(data_to_save, handle)

    
def analyze(Mass_cut, spec, sim):
    """
    Main analysis function.

    Parameters:
    - Mass_cut (float): Mass cut value.
    - spec (str): Specification identifier.
    - sim (str): Simulation identifier.
    """
    global L, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list
    file_path = "/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/Runs_bulk_all_sims/"+spec+"/"+sim
    data_store = {}  # Dictionary to store ngrid_data data
    halo_cat = np.loadtxt(halo_dir+spec+"/"+sim+"/output/halos/out_2.list") # Loading the full halo catalogue to loop over each halo
    mass_conditions = (halo_cat[:,20]>=Mass_cut)                
    halo_cat = halo_cat[mass_conditions]  # applying the mass condition
    pos_halos = halo_cat[:,8:11];
    vel_halos = halo_cat[:,11:14];
    masses = halo_cat[:,20];
    biases = bias.haloBias(masses, model = 'tinker10', z = 0.0, mdef = '200m')
    z = 0.0
    mdef = '200m'
    print("The analysis is being done for Mass_cut:"+f"{Mass_cut:.2e}");
    metadata = {'simulation': sim, 'boxsize': L, 'save_path': save_path, 'N_halos': np.shape(masses)[0], 'N_halos(M>M_cut)/M_tot': np.shape(masses)[0]/(np.shape(mass_conditions)[0]), 'Mass cut':f'{Mass_cut:.1e}', '<(v_h - v_bulk(halos).((bias_h + (mass)_h/(1.3e14))^0.85)* (<v(nu)> - v_h^i/2)>': '<y_{h-bulk(h)}.x(nuhi)>', '<(v_h - v_bulk(halos).((bias_h + (mass)_h/(1.3e14))^0.85)* (<v_bulk(nu)> - <v_h>/2))>': '<y_{h-bulk(h)}.x(nuh)>',
                    '<(bias_h + (mass)_h/(1.3e14))^0.85)* v_bulk(cnu)) * (bias_h + (mass)_h/(1.3e14))^0.85)* v_bulk(cnu))>':'<x(cnu)^2>' }
    for i in range(np.shape(pos_halos)[0]): # loop over halos
        if (i % 10000==0):
            if rank == 0:
                print("Analysis is done for ", i*100. / np.shape(pos_halos)[0], "% of the halos")
                
        for ngrid in ngrid_list:
            #####
            try:
                cdm_bulk_all, nu_bulk_all, halo_bulk_all = load_data_bulk(file_path, ngrid, sim, spec, Mass_cut) ### Loading all files for bulk velocities
                # print("Data loaded successfully.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
            #############
            coeff = ngrid / (L+0.1);
            x_index = int(np.floor(pos_halos[i, 0] * coeff))
            y_index = int(np.floor(pos_halos[i, 1] * coeff))
            z_index = int(np.floor(pos_halos[i, 2] * coeff))
            sub_box_index = (x_index, y_index, z_index)

            if tuple(sub_box_index) in halo_bulk_all['sub_box_data']:
                if sim != "0.0ev":
                    v_nu_cell = nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] ##nu_bulk_all['bulk_vel'][sub_box];
                else:
                    v_nu_cell = np.array([0.,0.,0.]);
                v_h_cell = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']  
                v_c_cell = cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']  
                ## two averages important in regression
                coefficent_cell = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_b_M']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] # Bulk <((bias_h + (mass)_h/(1.3e14))^0.85)> for all halos within the cell 
                b_M_v_h_cell = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_b_M_vel_h']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] # <(bias_h + (mass)_h/(1.3e14))^0.85 * v_h/2.> for all halos within the cell 
                v_cnu_cell = v_c_cell - v_nu_cell;
                v_nuh_cell = v_nu_cell - v_h_cell/2.;
                v_nuh2_i = v_nu_cell - vel_halos[i]/2.;
                v_ch_cell = v_c_cell - v_h_cell/2.;
                
                ############################################
                mass = masses[i]
                bias_h = biases[i]
                coefficent = (bias_h + (mass/(1.3*1.e14))**(0.85));
                ####
                x_minus_xavg_1 = (coefficent - coefficent_cell) * v_nuh_cell # x-<x>   --> # x^i = coefficent * <v_nu> - <v_h_cell>/2
                x_dot_y_1 = np.dot(vel_halos[i]  - v_h_cell, x_minus_xavg_1) # (y-<y>).(x-<x>) where <y> = 0  ----x^i = coefficent * <v_nu> - <v_h_cell>/2, y^i = vel_halos[i]  - v_h_cell
                #####
                x_minus_xavg_2 = (coefficent - coefficent_cell) * v_c_cell # x-<x>   --> # x^i = coefficent * <v_nu> - <v_h_cell>/2
                x_dot_y_2 = np.dot(vel_halos[i]  - v_h_cell, x_minus_xavg_2)  # y^i = coeff * v_c
                #####
                x_minus_xavg_3 = (coefficent - coefficent_cell) * v_h_cell
                x_dot_y_3 = np.dot(vel_halos[i]  - v_h_cell, x_minus_xavg_3) # y^i = coeff * v_h
                #####
                x_minus_xavg_4 = (coefficent - coefficent_cell) * v_ch_cell
                x_dot_y_4 = np.dot(vel_halos[i]  - v_h_cell, x_minus_xavg_4)
                #####
                x_minus_xavg_5 = (coefficent - coefficent_cell) * v_cnu_cell
                x_dot_y_5 = np.dot(vel_halos[i]  - v_h_cell, x_minus_xavg_5)
                #####
                x_avg_6 = coefficent_cell * v_nu_cell - b_M_v_h_cell/2.;
                x_minus_xavg_6 = coefficent * v_nuh2_i - x_avg_6
                x_dot_y_6 = np.dot(vel_halos[i]  - v_h_cell, x_minus_xavg_6)# correlation between v_h - <v_h> and (Coeff* (<v_nu> - v_h^i/2.)- <Coeff* (<v_nu> - v_h^i/2.)>
                #########
                x_dot_x_1 = np.dot(x_minus_xavg_1, x_minus_xavg_1) # computing (x - <x>)(x-<x>)
                x_dot_x_2 = np.dot(x_minus_xavg_2, x_minus_xavg_2)
                x_dot_x_3 = np.dot(x_minus_xavg_3, x_minus_xavg_3)
                x_dot_x_4 = np.dot(x_minus_xavg_4, x_minus_xavg_4)
                x_dot_x_5 = np.dot(x_minus_xavg_5, x_minus_xavg_5)
                x_dot_x_6 = np.dot(x_minus_xavg_6, x_minus_xavg_6)
                ####### Information to be stored
                infoxy1 = '<y_{h-bulk(h)}.x(nuh)>'
                infoxy2 = '<y_{h-bulk(h)}.x(c)>'
                infoxy3 = '<y_{h-bulk(h)}.x(h)>'
                infoxy4 = '<y_{h-bulk(h)}.x(ch)>'
                infoxy5 = '<y_{h-bulk(h)}.x(cnu)>'
                infoxy6 = '<y_{h-bulk(h)}.x(nuhi/2)>' # v_nu_cell - vel_halos[i]/2.;
                ###
                infoxx1 = '<x(nuh)^2>' # v_nu_cell - v_h_cell/2.;
                infoxx2 = '<x(c)^2>'
                infoxx3 = '<x(h)^2>'
                infoxx4 = '<x(ch)^2>' # v_c_cell - v_h_cell/2.;
                infoxx5 ='<x(cnu)^2>'
                infoxx6 = '<x(nuhi)^2>' # ((b_h^i + M_h^i )*(v_nu_cell - vel_halos[i]/2.) - <(b_h^i + M_h^i )*(v_nu_cell - vel_halos[i]/2.)>) ;
                if ngrid not in data_store:
                    data_store[ngrid] = {
                        infoxy1: 0.0,
                        infoxy2: 0.0,
                        infoxy3: 0.0,
                        infoxy4: 0.0,
                        infoxy5: 0.0,
                        infoxy6: 0.0,
                        infoxx1: 0.0,
                        infoxx2: 0.0,
                        infoxx3: 0.0,
                        infoxx4: 0.0,
                        infoxx5: 0.0,
                        infoxx6: 0.0,
                        'ngrid': ngrid,
                        'dx': L/ngrid,
                        'N': 0.0
                    }

                # Accumulate velocity and number of particles in the sub-box
                data_store[ngrid][infoxy1] += x_dot_y_1
                data_store[ngrid][infoxy2] += x_dot_y_2
                data_store[ngrid][infoxy3] += x_dot_y_3
                data_store[ngrid][infoxy4] += x_dot_y_4
                data_store[ngrid][infoxy5] += x_dot_y_5
                data_store[ngrid][infoxy6] += x_dot_y_6
                
                data_store[ngrid][infoxx1] += x_dot_x_1
                data_store[ngrid][infoxx2] += x_dot_x_2
                data_store[ngrid][infoxx3] += x_dot_x_3
                data_store[ngrid][infoxx4] += x_dot_x_4
                data_store[ngrid][infoxx5] += x_dot_x_5
                data_store[ngrid][infoxx6] += x_dot_x_6
                data_store[ngrid]['N'] += 1
                
    ### After looping over all particles we save the data!
    # Save the dataframe
    save_data(save_path, spec, sim, Mass_cut, metadata, data_store)
    print(sim,"  ,", spec, "  ,", f"{Mass_cut:.2e}"," Finished!")





#############
## Main part
#############
if __name__ == "__main__":    

    total_combinations = len(Mass_cuts) * len(specs) * len(sims)
    
    # Check if the number of cores matches the required number
    required_cores = total_combinations
    if size < required_cores:
        print(f"Warning: Number of cores ({size}) is less than required ({required_cores}).")
    elif size > required_cores:
        print(f"Warning: Number of cores ({size}) is larger than the required number ({required_cores}).")
        print("Only the required number of cores will be utilized.")
        size = required_cores  # Limit the number of cores to the total combinations
    
    # Calculate the number of combinations each process should handle
    combinations_per_process = total_combinations // size
    remainder = total_combinations % size
    
    # Determine the range of combinations each process should handle
    start_index = rank * combinations_per_process + min(rank, remainder)
    end_index = (rank + 1) * combinations_per_process + min(rank + 1, remainder)
    
    # Iterate over combinations assigned to this process
    for index in range(start_index, end_index):
        # Calculate the corresponding Mass_cut, spec, and sim
        Mass_cut_index = index % len(Mass_cuts)
        spec_index = (index // len(Mass_cuts)) % len(specs)
        sim_index = index // (len(Mass_cuts) * len(specs))
        Mass_cut = Mass_cuts[Mass_cut_index]
        spec = specs[spec_index]
        sim = sims[sim_index]

        # Perform analysis for the current combination
        analyze(Mass_cut, spec, sim)

    # Wait for all processes to complete
    comm.Barrier()