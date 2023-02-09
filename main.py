import sys
sys.path.append('/mn/stornext/u3/hassanif/projects/Neutrino_Niayesh/Analysis/nu_code/')
import library_snapshot
from library_snapshot import system_tools as tools
from library_snapshot import sim_analysis as analysis
from library_snapshot import parser as parse
import numpy as np
import pandas as pd
import pickle
import os


########################
# Parsing the settings
# #######################
parameters = parse.parse_parameter_file('./settings.ini')
number_smaples_ini = parameters['number_samples']
sim_size_ini = parameters['simulation']
boxsize = parameters['boxsize']
bulk_species_ini = parameters['bulk_species']
snap_num_ini = parameters['snapshot_number']
sim_type_ini = parameters['sim_type']
mass_limit = parameters['mass_limit']
data_load = tools.nested_dict(5, list)
save_path = parameters['save_path']
sim_path =parameters['sim_path']
ngrid_min =parameters['ngrid_min']
ngrid_max = parameters['ngrid_max']
ngrid_step = parameters['ngrid_step']
ngrid_ini = range(ngrid_min,ngrid_max,ngrid_step)

for sim_size in sim_size_ini: 
    obj = analysis.sim(boxsize)
    for bulk_species in bulk_species_ini: 
        for sim in sim_type_ini:
            for snap_num in snap_num_ini: 
                ######################
                ## particles read
                ######################
                if bulk_species=='cdm':
                    snapshot = sim_path + sim_size+'/'+sim+'/output/snap00'+str(snap_num)+'_cdm'
                    #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
                    ptype = [1]
                    cdm_pcl = obj.gadget_load(snapshot,ptype) # pos = arr[0], vel = arr[1];
                    data_load[sim_size][sim][bulk_species]['pos'] = cdm_pcl[0]; 
                    data_load[sim_size][sim][bulk_species]['vel'] = cdm_pcl[1];
                ######################
                ### Neutrinos read:
                ######################
                elif bulk_species=='nu':
                    print(" \t  Analyzing bulk species: {}".format(bulk_species))
                    print(" \t  Analyzing snapshot number: {}".format(snap_num))
                    print(" \t  Analyzing simulation: {}".format(sim))
                    print(" \t  bulk chosen to : {}".format(bulk_species))                        
                    ptype=[1];
                    snapshot =  sim_path +sim_size+'//'+sim+'/output/snap00'+str(snap_num)+'_ncdm0'
                    nu_pcl = obj.gadget_load(snapshot,ptype) # pos = arr[0], vel = arr[1];
                    data_load[sim_size][sim][bulk_species]['pos'] = nu_pcl[0]; 
                    data_load[sim_size][sim][bulk_species]['vel'] = nu_pcl[1];
                ######################
                ## halo read
                ######################
                halo_address = sim_path +sim_size+'/'+sim+'/output/halos/out_'+str(snap_num)+'.list'
                #pop 1
                halo_p1 = obj.halo_selection(halo_address, '+', mass_limit);
                data_load[sim_size][sim]['halo_p1']['pos'] = halo_p1[:,:3]
                data_load[sim_size][sim]['halo_p1']['vel'] = halo_p1[:,3:6]
                #pop 2
                halo_p2 = obj.halo_selection(halo_address, '-', mass_limit)
                data_load[sim_size][sim]['halo_p2']['pos'] = halo_p2[:,:3]
                data_load[sim_size][sim]['halo_p2']['vel'] = halo_p2[:,3:6]            
                #pop all
                halo_all = obj.halo_selection(halo_address, 'all', mass_limit)
                data_load[sim_size][sim]['halo']['pos'] = halo_all[:,:3]
                data_load[sim_size][sim]['halo']['vel'] = halo_all[:,3:6]

                print(" \t  Analyzing bulk species: {}".format(bulk_species))
                print(" \t  Analyzing snapshot number: {}".format(snap_num))
                print(" \t  Analyzing simulation: {}".format(sim))
                print(" \t  bulk chosen to : {}".format(bulk_species))
                print("\033[32m \t  number of  halos: "+str(np.shape(halo_all)[0])+", number of p1: "+str(np.shape(halo_p1)[0])+", number of p2: "+str(np.shape(halo_p2)[0])+" \033[0m")
                ##########
                ##########
                for number_smaples in number_smaples_ini:  
                    print("\033[34m \033[1m \n \n  ********* Analyzing sub-box count:{}".format(number_smaples)+" ********* \033[0m")
                    for ngrid in ngrid_ini: 
                        if(number_smaples>ngrid*ngrid*ngrid):
                            number_smaples_tmp = ngrid*ngrid*ngrid;
                            print("\033[31m \t  chosen number of samples is larger than total sub-boxes! number of sub-boxes changed to:"+str(number_smaples)+"\033[0m")
                        else:
                            number_smaples_tmp = number_smaples;
                        sub_box_lists = obj.Random_sub_box(ngrid,number_smaples_tmp);
                        ## Reseting the dictionary!
                        column_names = ['simulation','parameters','n_grid', 'sub_box_index','bulk_vel_halos_i', 'bulk_vel_cdm_i', 'vp1.v_b(cdm)', 'vp2.v_b(cdm)', 'vp1.v_b(halos)', 'vp2.v_b(halos)', 'v_h.v_b(cdm)','v_h.v_b(halos)']
                        df = pd.DataFrame(columns=column_names)
                        for i in range(number_smaples_tmp):
                            sub_box_index = sub_box_lists[i]
                            ##############################
                            ##### pcl velocities in each cell
                            ##############################
                            bulk_species = 'cdm'
                            pos_cdm = data_load[sim_size][sim][bulk_species]['pos']
                            vel_cdm = data_load[sim_size][sim][bulk_species]['vel']
                            cdm_vels = obj.vels_in_cell(ngrid, pos_cdm, vel_cdm, sub_box_index)# Gives the velocities of objects in a cell defined by sub-box index!

                            ##############################
                            ##### All halos velocities in each cell
                            ##############################
                            pos_halo_all = data_load[sim_size][sim]['halo']['pos']
                            vel_halo_all = data_load[sim_size][sim]['halo']['vel']
                            halos_vels_i = obj.vels_in_cell(ngrid, pos_halo_all, vel_halo_all, sub_box_index)# Gives the velocities of objects in a cell defined by sub-box index!

                            ##############################
                            ##### Selecting two populations of halos and computing each 
                            ##### population velocities in each cell! 
                            ##############################
                            pos_halo_p1 =  data_load[sim_size][sim]['halo_p1']['pos'] 
                            vel_halo_p1 =  data_load[sim_size][sim]['halo_p1']['vel']
                            pos_halo_p2 =  data_load[sim_size][sim]['halo_p2']['pos']
                            vel_halo_p2 =  data_load[sim_size][sim]['halo_p2']['vel']
                            halo_p2_vels_i = obj.vels_in_cell(ngrid, pos_halo_p2, vel_halo_p2, sub_box_index)
                            halo_p1_vels_i = obj.vels_in_cell(ngrid, pos_halo_p1, vel_halo_p1, sub_box_index)
                            if ( (np.shape(halo_p2_vels_i)[0]>0) and (np.shape(halo_p1_vels_i)[0])>0): # Only if we have halos from both population we do compute 

                                #############################
                                ### Bulk velocity computation
                                #############################
                                bulk_vel_halos_i = obj.velocity_bulk(ngrid, pos_halo_all, vel_halo_all, sub_box_index) # Bulk velocities using halos
                                bulk_vel_cdm_i = obj.velocity_bulk(ngrid, pos_cdm, vel_cdm, sub_box_index) # Bulk velocities using cdm

                                ##############################
                                ## Here in each cell we compute <vi^p1.v_b> and <v^p2.v_b>
                                ##############################
                                vdvb_p1_bulk_halos = obj.sum_vel_dot_vbulk(halo_p1_vels_i, bulk_vel_halos_i[0]); # This results in an array containing average of <v_i, v_b>, standard deviation and the number of halos in each cell!
                                vdvb_p2_bulk_halos = obj.sum_vel_dot_vbulk(halo_p2_vels_i, bulk_vel_halos_i[0]);

                                vdvb_p1_bulk_cdm = obj.sum_vel_dot_vbulk(halo_p1_vels_i, bulk_vel_cdm_i[0]); # This results in an array containing average of <v_i, v_b>, standard deviation and the number of halos in each cell!
                                vdvb_p2_bulk_cdm = obj.sum_vel_dot_vbulk(halo_p2_vels_i, bulk_vel_cdm_i[0]);

                                vdvb_all_bulk_cdm = obj.sum_vel_dot_vbulk(halos_vels_i, bulk_vel_cdm_i[0]); # This results in an array containing average of <v_i, v_b>, standard deviation and the number of halos in each cell!
                                vdvb_all_bulk_halos = obj.sum_vel_dot_vbulk(halos_vels_i, bulk_vel_halos_i[0]); # This results in an array containing average of <v_i, v_b>, standard deviation and the number of halos in each cell!

                                data_saved = {
                                    'simulation':sim,
                                    'parameters':parameters,
                                    'n_grid': ngrid,
                                    'sub_box_index': sub_box_index,
                                    'bulk_vel_halos_i': bulk_vel_halos_i,
                                    'bulk_vel_cdm_i': bulk_vel_cdm_i,
                                    'vp1.v_b(cdm)':vdvb_p1_bulk_cdm ,
                                    'vp2.v_b(cdm)':vdvb_p2_bulk_cdm ,
                                    'vp1.v_b(halos)':vdvb_p1_bulk_halos ,
                                    'vp2.v_b(halos)':vdvb_p2_bulk_halos ,
                                    'v_h.v_b(cdm)':vdvb_all_bulk_cdm ,
                                    'v_h.v_b(halos)':vdvb_all_bulk_halos 
                                    }
                                df = df.append(data_saved, ignore_index=True)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        with open(save_path+'/data_ngrid_'+str(ngrid)+'_sim_'+sim_size+'_'+sim+'_snap_'+str(snap_num)+'_Nsubboxes_'+str(number_smaples_tmp)+'.pickle', 'wb') as handle:
                            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print("\t \t *** number of grid: "+str(ngrid)+" is finished!")
                print("\033[34m \033[1m  \n \t The main loop is finished \033[0m")
                print("\033[34m \033[1m ************************************************* \033[0m")
