
import MAS_library as MASL
import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import readgadget
from collections import defaultdict
from mpi4py import MPI
import random
import psutil
import os
import pandas as pd
import pickle


def nested_dict(n, type):
    """
    This function creates a nested dictionary with a specified depth and default value type.

    Parameters:
        n (int): an integer representing the depth of the nested dictionary.
        type: the default value type for the nested dictionary.

    Returns:
        A nested dictionary with a depth of 'n' and default values of the specified type.
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))
    
data_load = nested_dict(5, list)

class massive_neutrinos:
    """
    Class for massive neutrino project
    """
    def __init__(self, boxsize):
        self.comm = MPI.COMM_WORLD
        self.boxsize = boxsize;
        return


    def norm(self, x):
        """ norm of a vector """
        return np.sqrt(sum(x_i*x_i for x_i in x));


    def dist(self, x, y):
        """Distance of two points"""
        return self.norm(np.subtract(x,y)) ;

    def statistics(self, x):
        """Given a vector it returns the average and std of the list! """
        return (np.average(x),np.std(x));

    def halo_selection(self, data_address, string, mass_limit, extra_columns=False):
        """ outputting a halo population obtained by a mass limit and whether we want larger or smaller halos.
            input: data (halo catalogue from Rockstar), a string being + or -, and mass limit in solar mass.
            output: A new catalogue with [x,y,z,v_x,v_y,v_z] and additional columns of[Rvir,M200b] of halo population if requested!
        """
        data = np.loadtxt(data_address);
        if (string == "+"):
            condition = data[:,20]> mass_limit;
            if extra_columns:
                halo_pop = np.zeros((np.shape(data[condition])[0],8))
            else:
                halo_pop = np.zeros((np.shape(data[condition])[0],6))
            for i in range(6):
                halo_pop[:,i] = data[condition][:,8+i]
            if (extra_columns):
                halo_pop[:,6] = data[condition][:,5] # R_vir
                halo_pop[:,7] = data[condition][:,20] # Mass
        elif (string == "-"):
            condition = data[:,20]<= mass_limit;
            if extra_columns:
                halo_pop = np.zeros((np.shape(data[condition])[0],8))
            else:
                halo_pop = np.zeros((np.shape(data[condition])[0],6))
            for i in range(6):
                halo_pop[:,i] = data[condition][:,8+i]
            if (extra_columns):
                halo_pop[:,6] = data[condition][:,5] # R_vir
                halo_pop[:,7] = data[condition][:,20] # Mass
 
        elif (string == "all"):
            if extra_columns:
                halo_pop = np.zeros((np.shape(data)[0],8))
            else:
                halo_pop = np.zeros((np.shape(data)[0],6))
            for i in range(6):
                halo_pop[:,i] = data[:,8+i]
            if (extra_columns):
                halo_pop[:,6] = data[:,5] # R_vir
                halo_pop[:,7] = data[:,20] # Mass

        return halo_pop;

    def sum_vel_dot_vbulk(self, vel, velocity_bulk):
        """computing v_i . v_bulk for halos/particle using the bulk velocity and returning the sum of all dot product and number of halos in that cell."""
        # product = np.zeros((np.shape(vel)[0]));
        # # for i in range(np.shape(vel)[0]):
        # #     product += np.dot(vel[i,:], velocity_bulk) # velocity bulk is an 3D array while vel =  N_halos x 3
        avg = np.mean(np.dot(vel, velocity_bulk))
        std = np.std(np.dot(vel, velocity_bulk))
        return [avg, std, np.shape(vel)[0]] # returns sum of all dot product and number of halos in that cell

    def vels_in_cell(self, ngrid, pos, vel, sub_box_index):
        boxsize = self.boxsize;
        """outputting the velocity of halos in the given sub-box """
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        vel_halos =vel[condition_tot];
        # pos_halos =pos[condition_tot]# For test you can output pos[vel_halos, pos_halos];
        return vel_halos

    def velocity_bulk(self, ngrid, pos,vel, sub_box_index):
        boxsize = self.boxsize;
        """computing the bulk velocity in a sub-box using the velocities of objects"""
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        vel_grids= np.array([np.average(vel[condition_tot,0]),np.average(vel[condition_tot,1]),np.average(vel[condition_tot,2])])
        sigma_vel_grids= np.array([np.std(vel[condition_tot,0]),np.std(vel[condition_tot,1]),np.std(vel[condition_tot,2])])
        return [vel_grids,sigma_vel_grids];


    def catalogue_header(self, address, column=-1):
        """Takes the address of the halo catalogue and reads the whole header by default. If the column is specified then just prints the corresponding column!"""
        f = open(address)
        header = f.readline()
        header_list = header.split(' ') #list of columns
        if column==-1:
            for i in range(np.shape(header_list)[0]):
                print(str(i)+":"+header_list[i])
        if column!=-1:
            print(str(column)+":"+header_list[column])

    def gadget_load(self, snapshot, ptype=[1]):
        """Takes the address of the a gadget 2 snapshot and returns the pos and velocity of particles!
           input: snapshot address, ptype -- > [1] for cdm and [2] for neutrinos!
        """
        pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
        vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
        # ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
        return np.array([pos,vel])


    # def Random_sub_box2(self, ngrid, number):
    #     """Takes the number of grids and return N randomly chosen sub-box indices!
    #     """
    #     res = np.zeros((number,3)).astype(int)
    #     for i in range(number):
    #         res[i] = random.sample(range(0, ngrid), 3)       
    #     return res
    
    def Random_sub_box(self, ngrid, number):
        """Takes the number of grids and return N randomly chosen sub-box indices!
        """
        res = np.zeros((number,3)).astype(int)
        for i in range(number):
            for j in range(3):
                res[i,j] = random.randint(0,ngrid-1)      
        return res


    def data_sub_boxes(self, sub_box_lists, ngrid, pos_pcl, vel_pcl, pos_p1, vel_p1, pos_p2, vel_p2):
        """
        Computing alpha and sigma_alpha for each sub-box -- returning all alphas for number of sub-boxes!
        Input: 
        - index of sub-boxes we want to consider!
        - ngrid :  Number of grids = the resolution we want to compute alpha over.
        - pos of particles we want to compute the bulk velocity from  -- N x 3 array containing x, y, z of pcls
        - vel of particles we want to compute the bulk velocity from -- N x 3 array containing v_x, v_y, v_z 
        - pos_p1: position of halo population 1, N_p1 x 3 array containing [x, y, z] for population 1
        - vel_p1: position of halo population 1, N_p1 x 3 array containing [vx, vy, vz] for population 1
        - pos_p2: position of halo population 2, N_p1 x 3 array containing [x_p2, y_p2, z_p2] for population 1
        - vel_p2: velocity of halo population 2, N_p1 x 3 array containing [vx_p2, vy_p2, vz_p2] for population 1
        output:
        - An array containing [index of sub-box, average of alpha, std of alpha]
        """
        boxsize = self.boxsize;
        data=[]
        for i in range (np.shape(sub_box_lists)[0]):
            sub_box_index = sub_box_lists[i]
            bulk_vel = self.velocity_bulk(ngrid, pos_pcl,vel_pcl, sub_box_index); # Bulk velocity using cdm/neutrino
            vels_p1 = self.vels_in_cell(ngrid, pos_p1, vel_p1, sub_box_index);# velocity of population 1
            vels_p2 = self.vels_in_cell(ngrid, pos_p2, vel_p2, sub_box_index); # velocity of population 2
            # ## computing v_b.v_i information for two populations!
            vdvb_p1 = self.sum_vel_dot_vbulk(vels_p1, bulk_vel); # This results in an array containing average of <v_i, v_b>, standard deviation and the number of halos in each cell!
            vdvb_p2 = self.sum_vel_dot_vbulk(vels_p2, bulk_vel);
            data.append([sub_box_index, vdvb_p1[0], vdvb_p2[0], vdvb_p1[1], vdvb_p2[1], vdvb_p1[2],  vdvb_p2[2]]);
            # [sub-box index, <v_i,v_b> population 1, <v_i,v_b> population 2, sigma v_i.v_b pop1, sigma v_i.v_b pop2, number of halos of pop1, number of halos of pop2]
        return data
    

    def Parallelized_alpha_sub_boxes(self, sub_box_lists, ngrid, pos_pcl, vel_pcl, pos_p1, vel_p1, pos_p2, vel_p2):
        """
        Parallelizing sub-boxes over number of CPUs and computing alpha and sigma_alpha for each sub-box -- returning all alphas for number of sub-boxes!
        Input: 
        - index of sub-boxes we want to consider!
        - ngrid :  Number of grids = the resolution we want to compute alpha over.
        - pos of particles we want to compute the bulk velocity from  -- N x 3 array containing x, y, z of pcls
        - vel of particles we want to compute the bulk velocity from -- N x 3 array containing v_x, v_y, v_z 
        - pos_p1: position of halo population 1, N_p1 x 3 array containing [x,y,z] for population 1
        - vel_p1: position of halo population 1, N_p1 x 3 array containing [vx,vy,vz] for population 1
        - pos_p2: position of halo population 2, N_p1 x 3 array containing [x_p2,y_p2,z_p2] for population 1
        - vel_p2: velocity of halo population 2, N_p1 x 3 array containing [vx_p2,vy_p2,vz_p2] for population 1
        output:
        - An array containing [index of sub-box, average of alpha, std of alpha]
        """
        comm = self.comm;
        boxsize = self.boxsize;
        rank = comm.Get_rank();
        size = comm.Get_size();
        t_start = MPI.Wtime()
        data=[]
        if rank == 0:
            a_row = sub_box_lists.shape[0]
            if a_row >= size:
                split = numpy.array_split(sub_box_lists, size, axis=0)
        else:
            split = None
        split = comm.scatter(split, root=0) # This scatters each part of vector over each CPU! The vector was initially produced based on number of CPUs.
        # split = numpy.dot(split, b)
        for i in range(np.shape(split)):
            sub_box_index = split[i];
            bulk_vel = velocity_bulk(ngrid, boxsize, pos,vel, sub_box_index); # Bulk velocity using cdm/neutrino
            vels_p1 = vels_in_cell(ngrid, boxsize, pos, vel, sub_box_index);# alpha for population 1
            vels_p2 = vels_in_cell(ngrid, boxsize, pos, vel, sub_box_index); # alpha for population 2

            alphas_p1 = alpha_list(vels_p1, bulk_vel);
            alphas_p2 = alpha_list(vels_p2, bulk_vel);

            data.append(sub_box_index, np.average(alpha_p1), np.average(alpha_p2), np.std(alpha_p1), np.std(alpha_p2));
            # data containing sub-nox index, average of alpha_pi and std of alphas!

        data = comm.gather(data, root=0)
        if rank == 0:
            result = numpy.vstack(data)
            t_diff_multi = MPI.Wtime() - t_start    ### Stop stopwatch ###
            print('Time taken one core:', t_diff_one, 'Time taken one multicore:', t_diff_multi)
        return result

def memory():
    psutil.virtual_memory()
    # you can convert that object to a dictionary 
    dict(psutil.virtual_memory()._asdict())
    pid = os.getpid()
    python_process = psutil.Process(pid)
    # you can have the percentage of used RAM
    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse, 'GB, percentage: ',psutil.virtual_memory().percent)
    
    
    
##################################
##################################
##################################
for number_smaples in [5, 25, 50, 100 , 250, 500, 1000]:  # How many sub-boxes

    print("****** computing for number of sub-boxes:"+str(number_smaples))
    for sim_size in ['L_512_N_256']:
        ##### parameters
        boxsize=512.0;
        mass_limit = 1.5*1.e13;
        obj = massive_neutrinos(boxsize);
        ### type of bulk species
        for bulk_species in ['cdm']:
            for snap_num in [3]: # the snapshot number start from high redshift
                for sim in ['Mn_0d6_nu_pcls','lcdm']:
                    print("******sim:"+sim+" is being read!**************")
                    ######################
                    ## particles read
                    ######################
                    if bulk_species=='cdm':
                        print("bulk chosen to cdm")
                        snapshot = './../Simulations///'+sim_size+'/'+sim+'/output/snap00'+str(snap_num)+'_cdm'
                        #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
                        ptype = [1]
                        cdm_pcl = obj.gadget_load(snapshot,ptype) # pos = arr[0], vel = arr[1];
                        data_load[sim_size][sim][bulk_species]['pos'] = cdm_pcl[0]; 
                        data_load[sim_size][sim][bulk_species]['vel'] = cdm_pcl[1];
                    ######################
                    ### Neutrinos read:
                    ######################
                    elif bulk_species=='nu':
                        print("bulk chosen to neutrino")
                        ptype=[1];
                        snapshot = './../Simulations//'+sim_size+'//'+sim+'/output/snap00'+str(snap_num)+'_ncdm0'
                        nu_pcl = obj.gadget_load(snapshot,ptype) # pos = arr[0], vel = arr[1];
                        data_load[sim_size][sim][bulk_species]['pos'] = nu_pcl[0]; 
                        data_load[sim_size][sim][bulk_species]['vel'] = nu_pcl[1];
                    ######################
                    ## halo read
                    ######################
                    halo_address ='./../Simulations/'+sim_size+'/'+sim+'/output/halos/out_'+str(snap_num)+'.list'
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
                    print("\033[32m number of halos: "+str(np.shape(halo_all)[0])+", number of p1: "+str(np.shape(halo_p1)[0])+", number of p2: "+str(np.shape(halo_p2)[0])+" \033[0m")
                    ##########

                    ##########
                    print("*****All files for sim:"+sim+" are read! Now we compute quantities in each sub-box ***********")

                    
                    for ngrid in range(1,40,2): 
                        if(number_smaples>ngrid*ngrid*ngrid):
                            number_smaples_tmp = ngrid*ngrid*ngrid;
                            print("\033[31m chosen number of samples is larger than total sub-boxes! number of sub-boxes changed to:"+str(number_smaples)+"\033[0m")
                        else:
                            number_smaples_tmp = number_smaples;
                        sub_box_lists = obj.Random_sub_box(ngrid,number_smaples_tmp);
                        ## Reseting the dictionary!
                        column_names = ['simulation','n_grid', 'sub_box_index','bulk_vel_halos_i', 'bulk_vel_cdm_i', 'vp1.v_b(cdm)', 'vp2.v_b(cdm)', 'vp1.v_b(halos)', 'vp2.v_b(halos)', 'v_h.v_b(cdm)','v_h.v_b(halos)']
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
                                    'ngrid': ngrid,
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
                        with open('./save_data/data_ngrid_'+str(ngrid)+'_sim_'+sim+'_Nsubboxes_'+str(number_smaples_tmp)+'.pickle', 'wb') as handle:
                            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print("*****ngrid:"+str(ngrid)+" is finished!***********")
                print("\033[34m *****simulation :"+sim+" is finished!*********** \033[0m")
