import numpy as np
import pandas as pd
import matplotlib as mpl
import pickle

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

do_the_analysis = True;


L= 1024.
sims= ["0.0ev", "0.15ev", "0.3ev" , "0.6ev"];
specs = ["L_1024_Ngrid_512", "L_1024_Ngrid_1024"]# specs = ["L_1024_Ngrid_512"]#, "L_1024_Ngrid_1024"]
halo_dir = "/mn/stornext/u3/hassanif/neutrino_niayesh/simulations/"
save_path = "./correlations_xy/"
ngrid_max = 75

########################################
### Loading all the N_grid bulk velocities at once: to prevent looping:
########################################
if do_the_analysis == True:
    data_ngrid = nested_dict(5, float) 
    for spec in specs:
        for sim in sims:
            #### Loading all N_grids at once!
            start_time = time.time()
            file_path = "./../Runs_bulk_velocities_10August2023//"+spec+"/"+sim
            for ngrid in np.arange(1,ngrid_max,1):
                cdm_path =file_path+"/cdm/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_snap002_cdm.pickle"
                if sim != "0.0ev":
                    nu_path =file_path+"/nu/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_snap002_ncdm0.pickle"
                else:
                    nu_path=""
                halo_all_path =file_path+"/halo_mass_1e12/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_halos_out_2.pickle"
                if (not os.path.exists(cdm_path)):
                    print(cdm_path," doesn't exist/ cdm!")
                if (not os.path.exists(nu_path)):
                    if sim != "0.0ev":
                        print(nu_path,"doesn't exist/ nu or it's lcdm case!")
                if (not os.path.exists(halo_all_path)):
                    print(halo_all_path,"doesn't exist/ halo!")
                if (os.path.exists(cdm_path) and os.path.exists(halo_all_path)):
                    print("Doing analysis for", ngrid, sim, spec)
                    data_ngrid[ngrid][sim][spec]['cdm_bulk_all'] = np.load(cdm_path, allow_pickle=True)
                    if sim != "0.0ev":
                        data_ngrid[ngrid][sim][spec]['nu_bulk_all'] = np.load(nu_path, allow_pickle=True)
                    data_ngrid[ngrid][sim][spec]['halo_bulk_all'] = np.load(halo_all_path, allow_pickle=True)
            end_time = time.time()
            print("For ",sim, " ",spec, "All n_grid data are loaded in {} seconds".format(end_time - start_time))
    end_time = time.time()
    print("All data for all sims are loaded in {} seconds".format(end_time - start_time))
    
    
### Now we loop over all halos and we compute the correlations

do_the_analysis = True;
Mass_cuts = [1.e12, 5.e13, 7.e13, 1.e14];

if do_the_analysis == True:
    for Mass_cut in Mass_cuts:
        for spec in specs:
            for sim in sims:
                start_time = time.time()
                data_store = {}  # Dictionary to store ngrid_data data
                halo_cat = np.loadtxt(halo_dir+spec+"/"+sim+"/output/halos/out_2.list")
                mass_conditions = (halo_cat[:,20]>=Mass_cut)                
                halo_cat = halo_cat[mass_conditions]
                pos_halos = halo_cat[:,8:11];
                vel_halos = halo_cat[:,11:14];
                masses = halo_cat[:,20];
                biases = bias.haloBias(masses, model = 'sheth01', z = 0.0, mdef = '200m')
                # Halo model:
                model = 'sheth01'
                z = 0.0
                mdef = '200m'
                print("The analysis is being done for Mass_cut:"+f"{Mass_cut:.2e}");
                data_store ={}
                metadata = {'simulation': sim, 'boxsize': L, 'save_path': save_path, 'N_halos': np.shape(masses)[0], 'N_halos(M>M_cut)/M_tot': np.shape(masses)[0]/(np.shape(mass_conditions)[0]), 'Mass cut':f'{Mass_cut:.1e}', '<(v_h - v_bulk(halos).((bias_h + (mass)_h/(1.3e14))^0.85)* v_bulk(nuh))>': '<y_{h-bulk(h)}.x(nuh)>', 
                            '<(bias_h + (mass)_h/(1.3e14))^0.85)* v_bulk(cnu)) * (bias_h + (mass)_h/(1.3e14))^0.85)* v_bulk(cnu))>':'<x(cnu)^2>' }
                for i in range(np.shape(pos_halos)[0]): # loop over halos
                    if (i % 1000==0):
                            print("Analysis is done for ", i*100. / np.shape(pos_halos)[0], "% of the halos")
                    for ngrid in np.arange(1,ngrid_max,1):
                        ################################
                        ############ Loop over halos: ######
                        ########################################
                        coeff = ngrid / (L+0.1);
                        x_index = int(np.floor(pos_halos[i, 0] * coeff))
                        y_index = int(np.floor(pos_halos[i, 1] * coeff))
                        z_index = int(np.floor(pos_halos[i, 2] * coeff))
                        sub_box_index = (x_index, y_index, z_index)
                        
                        cdm_bulk_all = data_ngrid[ngrid][sim][spec]['cdm_bulk_all'];
                        nu_bulk_all = data_ngrid[ngrid][sim][spec]['nu_bulk_all'];
                        halo_bulk_all = data_ngrid[ngrid][sim][spec]['halo_bulk_all'];

                        if tuple(sub_box_index) in halo_bulk_all['sub_box_data']:
                            if sim != "0.0ev":
                                v_nu_cell = nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] ##nu_bulk_all['bulk_vel'][sub_box];
                            else:
                                v_nu_cell = (0.,0.,0.);
                            v_h_cell = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']  
                            v_c_cell = cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']  
                            v_cnu_cell = v_c_cell - v_nu_cell;
                            v_nuh_cell = v_nu_cell - v_h_cell;
                            v_nuh2_i = v_nu_cell - vel_halos[i]/2.;
                            v_ch_cell = v_c_cell - v_h_cell;
                            ##### 
                            mass = masses[i]
                            bias_h = biases[i]
                            coefficent = (bias_h + (mass/(1.3*1.e14))**(0.85));

                            x_dot_y_1 = np.dot(vel_halos[i]  - v_h_cell, coefficent * v_nuh_cell) # 
                            x_dot_y_2 = np.dot(vel_halos[i]  - v_h_cell, coefficent * v_c_cell)
                            x_dot_y_3 = np.dot(vel_halos[i]  - v_h_cell, coefficent * v_h_cell)
                            x_dot_y_4 = np.dot(vel_halos[i]  - v_h_cell, coefficent * v_ch_cell)
                            x_dot_y_5 = np.dot(vel_halos[i]  - v_h_cell, coefficent * v_cnu_cell)
                            # correlation between v_h - <v_h> and <v_nu> - v_h/2.
                            x_dot_y_6 = np.dot(vel_halos[i]  - v_h_cell, coefficent * v_nuh2_i) 
                            
                            x_dot_y_1_n = np.dot(vel_halos[i]  - v_c_cell, coefficent * v_nuh_cell) # 
                            x_dot_y_2_n = np.dot(vel_halos[i]  - v_c_cell, coefficent * v_c_cell)
                            x_dot_y_3_n = np.dot(vel_halos[i]  - v_c_cell, coefficent * v_h_cell)
                            x_dot_y_4_n = np.dot(vel_halos[i]  - v_c_cell, coefficent * v_ch_cell)
                            x_dot_y_5_n = np.dot(vel_halos[i]  - v_c_cell, coefficent * v_cnu_cell)
                            #########
                            x_dot_x_1 = np.dot(coefficent * v_nuh_cell, coefficent * v_nuh_cell) 
                            x_dot_x_2 = np.dot(coefficent * v_c_cell, coefficent * v_c_cell)
                            x_dot_x_3 = np.dot(coefficent * v_h_cell, coefficent * v_h_cell)
                            x_dot_x_4 = np.dot(coefficent * v_ch_cell, coefficent * v_ch_cell) 
                            x_dot_x_5 = np.dot(coefficent * v_cnu_cell, coefficent * v_cnu_cell) 
                            x_dot_x_6 = np.dot(coefficent * v_nuh2_i, coefficent * v_nuh2_i)
                            #######

                            info1 = '<y_{h-bulk(h)}.x(nuh)>'
                            info2 = '<y_{h-bulk(h)}.x(c)>'
                            info3 = '<y_{h-bulk(h)}.x(h)>'
                            info4 = '<y_{h-bulk(h)}.x(ch)>'
                            info5 = '<y_{h-bulk(h)}.x(cnu)>'
                            info_new = '<y_{h-bulk(h)}.x(nuhi/2)>' # v_nu_cell - vel_halos[i]/2.;

                            info1_n = '<y_{h-bulk(cdm)}.x(nuh)>'
                            info2_n = '<y_{h-bulk(cdm)}.x(c)>'
                            info3_n = '<y_{h-bulk(cdm)}.x(h)>'
                            info4_n = '<y_{h-bulk(cdm)}.x(ch)>'
                            info5_n = '<y_{h-bulk(cdm)}.x(cnu)>'
                            ###
                            info6 = '<x(nuh)^2>'
                            info7 = '<x(c)^2>'
                            info8 = '<x(h)^2>'
                            info9 = '<x(ch)^2>'
                            info10 ='<x(cnu)^2>'
                            info_new_x = '<x(nuhi/2)^2>' # v_nu_cell - vel_halos[i]/2.;

                            if ngrid not in data_store:
                                data_store[ngrid] = {
                                    info1: 0.0,
                                    info2: 0.0,
                                    info3: 0.0,
                                    info4: 0.0,
                                    info5: 0.0,
                                    info_new: 0.0,
                                    info1_n: 0.0,
                                    info2_n: 0.0,
                                    info3_n: 0.0,
                                    info4_n: 0.0,
                                    info5_n: 0.0,
                                    info6: 0.0,
                                    info7: 0.0,
                                    info8: 0.0,
                                    info9: 0.0,
                                    info10: 0.0,
                                    info_new_x: 0.0,
                                    'ngrid': ngrid,
                                    'dx': L/ngrid,
                                    'N': 0.0
                                }

                            # Accumulate velocity and number of particles in the sub-box
                            data_store[ngrid][info1] += x_dot_y_1
                            data_store[ngrid][info2] += x_dot_y_2
                            data_store[ngrid][info3] += x_dot_y_3
                            data_store[ngrid][info4] += x_dot_y_4
                            data_store[ngrid][info5] += x_dot_y_5
                            data_store[ngrid][info_new] += x_dot_y_6

                            data_store[ngrid][info1_n] += x_dot_y_1_n
                            data_store[ngrid][info2_n] += x_dot_y_2_n
                            data_store[ngrid][info3_n] += x_dot_y_3_n
                            data_store[ngrid][info4_n] += x_dot_y_4_n
                            data_store[ngrid][info5_n] += x_dot_y_5_n
                            
                            data_store[ngrid][info6] += x_dot_x_1
                            data_store[ngrid][info7] += x_dot_x_2
                            data_store[ngrid][info8] += x_dot_x_3
                            data_store[ngrid][info9] += x_dot_x_4
                            data_store[ngrid][info10] += x_dot_x_5
                            data_store[ngrid][info_new_x] += x_dot_x_6
                            data_store[ngrid]['N'] += 1
                
                ### After looping over all particles we save the data!
                # Save the dataframe
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                data_to_save = {
                    'metadata': metadata,
                    'data': data_store
                }
                with open(save_path+'/data_correlations_'+spec+'_sim_'+sim+'_mass_'+f'{Mass_cut:.1e}'+'.pickle', 'wb') as handle:
                    pickle.dump(data_to_save, handle)
                end_time = time.time()
                print(sim,"  ,", spec, "  "," Finished!")
                print("The code took {} seconds to run.".format(end_time - start_time))
            