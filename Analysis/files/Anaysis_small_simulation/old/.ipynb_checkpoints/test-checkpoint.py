import MAS_library as MASL
import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import readgadget
from collections import defaultdict
from mpi4py import MPI
import random

class massive_neutrinos:
    """
    Class for massive neutrino project
    """
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        return
    
    def nested_dict(self, n, type):
        """ Definition of nested dictionary """
        if n == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda: nested_dict(n-1, type))
        
    def norm(self, x):
        """ norm of a vector """
        return np.sqrt(sum(x_i*x_i for x_i in x));

    def dot(self, x, y):
        """Dot product of two vectors"""
        return sum(x_i*y_i for x_i, y_i in zip(x, y));
    
    def dist(self, x, y):
        """Distance of two points"""
        return self.norm(np.subtract(x,y)) ;
    
    def statistics(x):
        """Given a vector it returns the average and std of the list! """
        return (np.average(x),np.std(x));
    
    def halo_selection(self, data, string='all', mass_limit): 
        """ outputting a halo population obtained by a mass limit and whether we want larger or smaller halos.
            input: data (halo catalogue from Rockstar), a string being + or -, and mass limit in solar mass.
            output: A new catalogue with [x,y,z,v_x,v_y,v_z,Rvir,M200b] of halo population
        """
        if (string == "+"):
            condition = data[:,20]> mass_limit;
            halo_pop = np.zeros((np.shape(data[condition])[0],8))
            for i in range(6):
                halo_pop[:,i] = data[condition][:,8+i]
            halo_pop[:,6] = data[condition][:,5] # R_vir
            halo_pop[:,7] = data[condition][:,20] # Mass
        elif (string == "-"):
            condition = data[:,20]<= mass_limit;
            halo_pop = np.zeros((np.shape(data[condition])[0],8))
            for i in range(6):
                halo_pop[:,i] = data[condition][:,8+i]
            halo_pop[:,6] = data[condition][:,5] # R_vir
            halo_pop[:,7] = data[condition][:,20] # Mass
        else:
            for i in range(6):
                halo_pop[:,i] = data[:,8+i]
            halo_pop[:,6] = data[:,5] # R_vir
            halo_pop[:,7] = data[:,20] # Mass
        return halo_pop;
    
    def alpha_list(vel, velocity_bulk):
        """computing alpha for halos/particle using the bulk velocity. The output is an array containing velocities."""
        alpha = np.zeros((np.shape(vel)[0]));
        for i in range(np.shape(vel)[0]):
            alpha[i] = dot(vel[i,:], velocity_bulk)/norm(velocity_bulk)
        return alpha;

    def vels_in_cell(ngrid, boxsize, pos,vel, sub_box_index): 
        """outputting the velocity of halos in the give sub-box """
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        vel_halos =vel[condition_tot];
        # pos_halos =pos[condition_tot]# For test you can output pos[vel_halos, pos_halos];
        return vel_halos 

    def velocity_bulk(ngrid, boxsize, pos,vel, sub_box_index):
        """computing the bulk velocity in a sub-box using the velocities of objects"""
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        vel_grids= np.array([np.average(vel[condition_tot,0]),np.average(vel[condition_tot,1]),np.average(vel[condition_tot,2])])
        sigma_vel_grids= np.array([np.std(vel[condition_tot,0]),np.std(vel[condition_tot,1]),np.std(vel[condition_tot,2])])
        return [vel_grids,sigma_vel_grids];   


    def catalogue_header(address, column=-1):
        """Takes the address of the halo catalogue and reads the whole header by default. If the column is specified then just prints the corresponding column!"""
        f = open(address)
        header = f.readline()
        header_list = header.split(' ') #list of columns
        if column==-1:
            for i in range(np.shape(header_list)[0]):
                print(str(i)+":"+header_list[i])
        if column!=-1:
            print(str(column)+":"+header_list[column])

    def gadget_load(snapshot, ptype=[1]):
        """Takes the address of the a gadget 2 snapshot and returns the pos and velocity of particles!
           input: snapshot address, ptype -- > [1] for cdm and [2] for neutrinos! 
        """
        pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
        vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
        # ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
        return np.array([pos,vel])
    
    
    def Random_sub_box(ngrid, number):
        """Takes the number of grids and return N randomly chosen sub-box indices! 
        """
        res = np.transpose([random.sample(range(0, ngrid), number),random.sample(range(0, ngrid), number),random.sample(range(0, ngrid), number)])
        return res
    
    
    def Parallelization_sub_boxes(self, sub_box_lists, pos_pcl, vel_pcl, halo_pop1, halo_pop2):
        """
        Parallelizing sub-boxes over number of CPUs and computing alpha and sigma_alpha for each sub-box -- returning all alphas for number of sub-boxes!
        """
        comm = self.comm;
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
            bulk_vel = velocity_bulk(ngrid, boxsize, pos,vel, sub_box_index);
            vels = vels_in_cell(ngrid, boxsize, pos,vel, sub_box_index);
            alphas = alpha_list(vel, velocity_bulk);
            data.append(splie, np.average(alpha), np.std(alpha));
            
        data = comm.gather(data, root=0)
        if rank == 0:
            result = numpy.vstack(data)
    t_diff_multi = MPI.Wtime() - t_start    ### Stop stopwatch ###
    if rank == 0 :
        print('Time taken one core:', t_diff_one, 'Time taken one multicore:', t_diff_multi)
        return result