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

## Here we get the data from both simulations and we compute $v_{cdm}$ - $v_{\nu}$ and then we compute different quantities including $v_{\nu} . v_{cdm}$,  $v_{\nu} . v_{cdm}$,  $v_{\nu} . v_{h}$ and different combinations to understand the correlation between different quantities

do_the_analysis = True;


L= 1024.
sims= ["0.0ev", "0.15ev", "0.3ev" , "0.6ev"];
specs = ["L_1024_Ngrid_512", "L_1024_Ngrid_1024"]# specs = ["L_1024_Ngrid_512"]#, "L_1024_Ngrid_1024"]
if do_the_analysis == True:
    for spec in specs:
        for sim in sims:
            df = pd.DataFrame()
            file_path = "./../Runs_bulk_velocities_10August2023//"+spec+"/"+sim
            for ngrid in np.arange(1,75,1):
                cdm_path =file_path+"/cdm/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_snap002_cdm.pickle"
                if sim != "0.0ev":
                    nu_path =file_path+"/nu/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_snap002_ncdm0.pickle"
                else:
                    nu_path=""
                halo_all_path =file_path+"/halo_mass_1e12/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_halos_out_2.pickle"
                halo_mass1_path =file_path+"/halo_mass_5e13/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_halos_out_2.pickle"
                halo_mass2_path =file_path+"/halo_mass_7e13/output/data_ngrid_"+str(ngrid)+"_sim_"+sim+"_"+spec+"_halos_out_2.pickle"
                
                if (not os.path.exists(cdm_path)):
                    print(cdm_path," doesn't exist/ cdm!")
                if (not os.path.exists(nu_path)):
                    print(nu_path,"doesn't exist/ nu or it's lcdm case!")
                if (not os.path.exists(halo_all_path)):
                    print(halo_all_path,"doesn't exist/ halo!")
                if (not os.path.exists(halo_mass1_path)):
                    print(halo_mass1_path,"doesn't exist/ halo!")
                if (not os.path.exists(halo_mass2_path)):
                    print(halo_mass2_path,"doesn't exist/ halo!")
                if (os.path.exists(cdm_path) and os.path.exists(halo_all_path) and os.path.exists(halo_mass1_path) and os.path.exists(halo_mass2_path)):
                    print("Doing analysis for", ngrid, sim, spec)
                    cdm_bulk_all = np.load(cdm_path, allow_pickle=True)
                    if sim != "0.0ev":
                        nu_bulk_all = np.load(nu_path, allow_pickle=True)
                    halo_bulk_all = np.load(halo_all_path, allow_pickle=True)
                    halo_5e13_bulk_all = np.load(halo_mass1_path, allow_pickle=True)
                    halo_7e13_bulk_all = np.load(halo_mass2_path, allow_pickle=True)
                    ##########
                    v_ch_dot_v_nuc_print = 0.
                    v_cnu_dot_v_cnu_print = 0.
                    v_nu_dot_v_nu_print = 0.
                    v_c_dot_v_nu_print = 0.
                    v_c_dot_v_c_print = 0.
                    ###### All halos prints #####
                    v_nu_dot_v_h_print = 0
                    v_c_dot_v_h_print = 0 
                    v_ch_dot_v_nuc_print = 0 
                    v_nuh_dot_v_nuc_print = 0 
                    v_nuh_dot_v_ch_print = 0 
                    v_h_dot_v_nuc_print = 0 
                    v_h_dot_v_h_print = 0 
                    v_ch_dot_v_ch_print = 0 
                    v_nuh_dot_v_nuh_print = 0 
                    ###### Mass cut 1 prints #####
                    v_nu_dot_v_h_m5e13_print = 0
                    v_c_dot_v_h_m5e13_print = 0
                    v_ch_dot_v_nuc_m5e13_print  = 0
                    v_nuh_dot_v_nuc_m5e13_print = 0
                    v_nuh_dot_v_ch_m5e13_print = 0
                    v_h_dot_v_nuc_m5e13_print = 0
                    v_h_dot_v_h_m5e13_print = 0
                    v_ch_dot_v_ch_m5e13_print = 0
                    v_nuh_dot_v_nuh_m5e13_print = 0
                    ###### Mass cut 2 prints #####
                    v_nu_dot_v_h_m7e13_print = 0
                    v_c_dot_v_h_m7e13_print = 0
                    v_ch_dot_v_nuc_m7e13_print  = 0
                    v_nuh_dot_v_nuc_m7e13_print = 0
                    v_nuh_dot_v_ch_m7e13_print = 0
                    v_h_dot_v_nuc_m7e13_print = 0
                    v_h_dot_v_h_m7e13_print = 0
                    v_ch_dot_v_ch_m7e13_print = 0
                    v_nuh_dot_v_nuh_m7e13_print = 0
                    
                    ################################
                    ############ Loop over halos: ######
                    ########################################
                    all_sub_boxes = list(product(range(ngrid), repeat=3))
                    for sub_box_index in all_sub_boxes:
                        #############
                        v_c_cell = cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/cdm_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']  #cdm_bulk_all['bulk_vel'][sub_box];
                        if sim != "0.0ev":
                            v_nu_cell = nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/nu_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] ##nu_bulk_all['bulk_vel'][sub_box];
                        else:
                            v_nu_cell = (0.,0.,0.);
                        v_cnu_cell = v_c_cell - v_nu_cell;
                        #############
                        v_c_dot_v_nu_print += np.dot(v_nu_cell, v_c_cell)
                        v_cnu_dot_v_cnu_print += np.dot(v_cnu_cell, v_cnu_cell)
                        v_c_dot_v_c_print += np.dot(v_c_cell, v_c_cell)
                        v_nu_dot_v_nu_print += np.dot(v_nu_cell, v_nu_cell)
                       
                        ##############  
                        #### All halos    
                        ##############
                        if tuple(sub_box_index) in halo_bulk_all['sub_box_data']:
                            v_h_cell = halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/halo_bulk_all['sub_box_data'][tuple(sub_box_index)]['N']  ### halo_bulk_all['bulk_vel'][sub_box];
                            v_nuh_cell = v_nu_cell - v_h_cell;
                            v_ch_cell = v_c_cell - v_h_cell;
                            ## dot products:
                            v_nu_dot_v_h = np.dot(v_nu_cell, v_h_cell) # v_nu.v_h all vectors!
                            v_c_dot_v_h = np.dot(v_c_cell, v_h_cell) # v_c .v_h all vectors!
                            v_ch_dot_v_nuc = np.dot(v_ch_cell, v_cnu_cell) # (v_c - v_h).(v_c - v_nu) all vectors!
                            v_nuh_dot_v_nuc = np.dot(v_nuh_cell, v_cnu_cell) # (v_nu - v_h).(v_c - v_nu) all vectors!
                            v_nuh_dot_v_ch = np.dot(v_nuh_cell, v_ch_cell) # (v_nu - v_h).(v_c - v_nu) all vectors!
                            v_h_dot_v_nuc = np.dot(v_cnu_cell, v_h_cell) # v_h.(v_c - v_nu) all vectors!
                                ### norms
                            v_h_dot_v_h = np.dot(v_h_cell, v_h_cell) # v_h.v_h all vectors!
                            v_ch_dot_v_ch = np.dot(v_ch_cell, v_ch_cell) #(v_c - v_h).(v_c - v_h) all vectors!
                            v_nuh_dot_v_nuh = np.dot(v_nuh_cell, v_nuh_cell) # (v_nu - v_h).(v_nu - v_h) all vectors!
                            if v_h_dot_v_h != 0 and not math.isnan(v_h_dot_v_nuc/v_h_dot_v_h):
                                v_nu_dot_v_h_print += v_nu_dot_v_h
                                v_c_dot_v_h_print += v_c_dot_v_h
                                v_ch_dot_v_nuc_print += v_ch_dot_v_nuc
                                v_nuh_dot_v_nuc_print += v_nuh_dot_v_nuc
                                v_nuh_dot_v_ch_print += v_nuh_dot_v_ch
                                v_h_dot_v_nuc_print += v_h_dot_v_nuc
                                v_h_dot_v_h_print += v_h_dot_v_h
                                v_ch_dot_v_ch_print += v_ch_dot_v_ch
                                v_nuh_dot_v_nuh_print += v_nuh_dot_v_nuh
                            
                        ##############  
                        #### Mass cut    
                        ##############
                        if tuple(sub_box_index) in halo_5e13_bulk_all['sub_box_data']:
                            v_h_cell = halo_5e13_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/halo_5e13_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] ## halo_5e13_bulk_all['bulk_vel'][sub_box];
                            v_nuh_cell = v_nu_cell - v_h_cell;
                            v_ch_cell = v_c_cell - v_h_cell;
                            ## dot products:
                            v_nu_dot_v_h = np.dot(v_nu_cell, v_h_cell) # v_nu.v_h all vectors!
                            v_c_dot_v_h = np.dot(v_c_cell, v_h_cell) # v_c .v_h all vectors!
                            v_ch_dot_v_nuc = np.dot(v_ch_cell, v_cnu_cell) # (v_c - v_h).(v_c - v_nu) all vectors!
                            v_nuh_dot_v_nuc = np.dot(v_nuh_cell, v_cnu_cell) # (v_nu - v_h).(v_c - v_nu) all vectors!
                            v_nuh_dot_v_ch = np.dot(v_nuh_cell, v_ch_cell) # (v_nu - v_h).(v_c - v_nu) all vectors!
                            v_h_dot_v_nuc = np.dot(v_cnu_cell, v_h_cell) # v_h.(v_c - v_nu) all vectors!
                                ### norms
                            v_h_dot_v_h = np.dot(v_h_cell, v_h_cell) # v_h.v_h all vectors!
                            v_ch_dot_v_ch = np.dot(v_ch_cell, v_ch_cell) #(v_c - v_h).(v_c - v_h) all vectors!
                            v_nuh_dot_v_nuh = np.dot(v_nuh_cell, v_nuh_cell) # (v_nu - v_h).(v_nu - v_h) all vectors!
                            if v_h_dot_v_h != 0 and not math.isnan(v_h_dot_v_nuc/v_h_dot_v_h):
                                v_nu_dot_v_h_m5e13_print += v_nu_dot_v_h
                                v_c_dot_v_h_m5e13_print += v_c_dot_v_h
                                v_ch_dot_v_nuc_m5e13_print += v_ch_dot_v_nuc
                                v_nuh_dot_v_nuc_m5e13_print += v_nuh_dot_v_nuc
                                v_nuh_dot_v_ch_m5e13_print += v_nuh_dot_v_ch
                                v_h_dot_v_nuc_m5e13_print += v_h_dot_v_nuc
                                v_h_dot_v_h_m5e13_print += v_h_dot_v_h
                                v_ch_dot_v_ch_m5e13_print += v_ch_dot_v_ch
                                v_nuh_dot_v_nuh_m5e13_print += v_nuh_dot_v_nuh

                        ##############  
                        #### Mass cut 2    
                        ############## 
                        if tuple(sub_box_index) in halo_7e13_bulk_all['sub_box_data']:
                            v_h_cell = halo_7e13_bulk_all['sub_box_data'][tuple(sub_box_index)]['sum_bulk_vel']/halo_7e13_bulk_all['sub_box_data'][tuple(sub_box_index)]['N'] ### halo_7e13_bulk_all['bulk_vel'][sub_box];
                            v_h_dot_v_nuc = np.dot(v_cnu_cell, v_h_cell) # v_h.(v_c - v_nu) all vectors!
                            v_h_dot_v_h = np.dot(v_h_cell, v_h_cell)
                            v_nuh_cell = v_nu_cell - v_h_cell;
                            v_ch_cell = v_c_cell - v_h_cell;
                            ## dot products:
                            v_nu_dot_v_h = np.dot(v_nu_cell, v_h_cell) # v_nu.v_h all vectors!
                            v_c_dot_v_h = np.dot(v_c_cell, v_h_cell) # v_c .v_h all vectors!
                            v_ch_dot_v_nuc = np.dot(v_ch_cell, v_cnu_cell) # (v_c - v_h).(v_c - v_nu) all vectors!
                            v_nuh_dot_v_nuc = np.dot(v_nuh_cell, v_cnu_cell) # (v_nu - v_h).(v_c - v_nu) all vectors!
                            v_nuh_dot_v_ch = np.dot(v_nuh_cell, v_ch_cell) # (v_nu - v_h).(v_c - v_nu) all vectors!
                            v_h_dot_v_nuc = np.dot(v_cnu_cell, v_h_cell) # v_h.(v_c - v_nu) all vectors!
                                ### norms
                            v_h_dot_v_h = np.dot(v_h_cell, v_h_cell) # v_h.v_h all vectors!
                            v_ch_dot_v_ch = np.dot(v_ch_cell, v_ch_cell) #(v_c - v_h).(v_c - v_h) all vectors!
                            v_nuh_dot_v_nuh = np.dot(v_nuh_cell, v_nuh_cell) # (v_nu - v_h).(v_nu - v_h) all vectors!
                            if v_h_dot_v_h != 0 and not math.isnan(v_h_dot_v_nuc/v_h_dot_v_h):
                                v_nu_dot_v_h_m7e13_print += v_nu_dot_v_h
                                v_c_dot_v_h_m7e13_print += v_c_dot_v_h
                                v_ch_dot_v_nuc_m7e13_print += v_ch_dot_v_nuc
                                v_nuh_dot_v_nuc_m7e13_print += v_nuh_dot_v_nuc
                                v_nuh_dot_v_ch_m7e13_print += v_nuh_dot_v_ch
                                v_h_dot_v_nuc_m7e13_print += v_h_dot_v_nuc
                                v_h_dot_v_h_m7e13_print += v_h_dot_v_h
                                v_ch_dot_v_ch_m7e13_print += v_ch_dot_v_ch
                                v_nuh_dot_v_nuh_m7e13_print += v_nuh_dot_v_nuh


                        # create a dictionary with the values
                    row_dict = {'sim': sim, 'n_grid': ngrid, 'dx': L/ngrid, 'v_cnu.v_cnu':v_cnu_dot_v_cnu_print/ngrid**3, 'v_nu.v_c': v_c_dot_v_nu_print/ngrid**3,
                            'v_c.v_c':v_c_dot_v_c_print/ngrid**3, 'v_nu.v_nu':v_nu_dot_v_nu_print/ngrid**3, 'v_h.v_h':v_h_dot_v_h_print/ngrid**3,
                               'v_h.v_h(M>5.e13)':v_h_dot_v_h_m5e13_print/ngrid**3, 'v_h.v_h(M>7.e13)':v_h_dot_v_h_m7e13_print/ngrid**3,
                                ########
                                'v_nu.v_h':v_nu_dot_v_h_print/ngrid**3, 'v_c.v_h':v_c_dot_v_h_print/ngrid**3,
                                'v_ch.v_nuc':v_ch_dot_v_nuc_print/ngrid**3, 'v_nuh.v_nuc':v_nuh_dot_v_nuc_print/ngrid**3,
                                'v_nuh.v_ch':v_nuh_dot_v_ch_print/ngrid**3, 'v_h.v_nuc':v_h_dot_v_nuc_print/ngrid**3,
                                'v_ch.v_ch':v_ch_dot_v_ch_print/ngrid**3, 'v_nuh.v_nuh':v_nuh_dot_v_nuh_print/ngrid**3,
                                ########
                                'v_nu.v_h(M>5.e13)':v_nu_dot_v_h_m5e13_print/ngrid**3, 'v_c.v_h(M>5.e13)':v_c_dot_v_h_m5e13_print/ngrid**3,
                                'v_ch.v_nuc(M>5.e13)':v_ch_dot_v_nuc_m5e13_print/ngrid**3, 'v_nuh.v_nuc(M>5.e13)':v_nuh_dot_v_nuc_m5e13_print/ngrid**3,
                                'v_nuh.v_ch(M>5.e13)':v_nuh_dot_v_ch_m5e13_print/ngrid**3,'v_h.v_nuc(M>5.e13)':v_h_dot_v_nuc_m5e13_print/ngrid**3,
                                'v_ch.v_ch(M>5.e13)':v_ch_dot_v_ch_m5e13_print/ngrid**3, 'v_nuh.v_nuh(M>5.e13)':v_nuh_dot_v_nuh_m5e13_print/ngrid**3,
                                ########
                                'v_nu.v_h(M>7.e13)':v_nu_dot_v_h_m7e13_print/ngrid**3, 'v_c.v_h(M>7.e13)':v_c_dot_v_h_m7e13_print/ngrid**3,
                                'v_ch.v_nuc(M>7.e13)':v_ch_dot_v_nuc_m7e13_print/ngrid**3, 'v_nuh.v_nuc(M>7.e13)':v_nuh_dot_v_nuc_m7e13_print/ngrid**3,
                                'v_nuh.v_ch(M>7.e13)':v_nuh_dot_v_ch_m7e13_print/ngrid**3,'v_h.v_nuc(M>7.e13)':v_h_dot_v_nuc_m7e13_print/ngrid**3,
                                'v_ch.v_ch(M>7.e13)':v_ch_dot_v_ch_m7e13_print/ngrid**3, 'v_nuh.v_nuh(M>7.e13)':v_nuh_dot_v_nuh_m7e13_print/ngrid**3,
                                 'N sub-boxes we have halos(all)': len(halo_bulk_all['sub_box_data']), 'N sub-boxes we have halos(M>5e13)': len(halo_5e13_bulk_all['sub_box_data']),
                                'N sub-boxes we have halos(M>7e13)':len(halo_7e13_bulk_all['sub_box_data'])}                # append the row to the dataframe
                    df = df.append(row_dict, ignore_index=True)

            df.to_csv('./bulk_velocity_analysis/bulk_vel_sim_'+sim+'_'+spec+'.csv', index=False);
