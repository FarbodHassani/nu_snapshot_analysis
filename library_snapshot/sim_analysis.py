"""
This module contains the sim_analysis class, which provides functionality for
working with simulation data. 
"""

import MAS_library as MASL
import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import readgadget
from mpi4py import MPI
import random
import pandas as pd
import pickle

#

class sim:
    """
    This class provides a set of tools for working with massive neutrino simulation data. It includes methods for computing the norm of a vector, the distance between two points, and statistics on a vector. It also includes a halo_selection method for selecting halos based on a mass limit and returning a new catalogue, and a sum_vel_dot_vbulk method for computing the sum of the dot product of the velocity of halos/particles with the bulk velocity. It is initialized with a boxsize attribute, and it also include a comm attribute that holds the MPI communicator.
    """
    def __init__(self, boxsize):
        """
        Initialize the class with MPI communication object and boxsize.

        Parameters:
            boxsize (float): size of the simulation box.

        Returns:
            None
        """
        self.comm = MPI.COMM_WORLD
        self.boxsize = boxsize;
        return


    def norm(self, x):
        """
        Calculate the norm of a given vector.
        
        Parameters:
            x (list or numpy array): vector for which the norm is to be calculated.
        
        Returns:
            float: norm of the vector x.
        """
        return np.sqrt(sum(x_i*x_i for x_i in x));


    def dist(self, x, y):
        """
        Calculate the distance between two points.
        
        Parameters:
            x (list or numpy array): first point.
            y (list or numpy array): second point.
        
        Returns:
            float: distance between points x and y.
        """
        return self.norm(np.subtract(x,y)) ;

    def statistics(self, x):
        """
    Given a vector, returns the average and standard deviation of the values in the list.
    
    Parameters:
        x (list or array): a vector containing numerical values
        
    Returns:
        A tuple of the form (average, standard deviation)
        """
        return (np.average(x),np.std(x));

    def halo_selection(self, data_address, string, mass_limit, Remove_subhalo="no",  extra_columns=False):
        """ 
        This function filters a halo catalogue from Rockstar based on a given mass limit and a string indicating whether to select larger or smaller halos.
Input:
- data_address: file path to the halo catalogue
- string: "+" for selecting halos above the mass limit, "-" for selecting halos below the mass limit, "all" for selecting all halos
- mass_limit: the mass limit for selecting halos
- extra_columns (default=False): flag for whether to include additional columns of [Rvir,M200b] in the output
Output:
- A new catalogue with [x,y,z,v_x,v_y,v_z] and additional columns of[Rvir,M200b] of halo population if requested
        """
        data = np.loadtxt(data_address);
        if (Remove_subhalo=='yes'):
            data = data[data[:,41]==-1]; # Only choose parent halos
        if (string == "+"):
            condition = data[:,20]> mass_limit;
            if extra_columns == True:
                halo_pop = np.zeros((np.shape(data[condition])[0],8))
            else:
                halo_pop = np.zeros((np.shape(data[condition])[0],6))
            for i in range(6):
                halo_pop[:,i] = data[condition][:,8+i]
            if extra_columns == True:
                halo_pop[:,6] = data[condition][:,5] # R_vir
                halo_pop[:,7] = data[condition][:,20] # Mass
        elif (string == "-"):
            condition = data[:,20]<= mass_limit;
            if extra_columns == True:
                halo_pop = np.zeros((np.shape(data[condition])[0],8))
            else:
                halo_pop = np.zeros((np.shape(data[condition])[0],6))
            for i in range(6):
                halo_pop[:,i] = data[condition][:,8+i]
            if (extra_columns):
                halo_pop[:,6] = data[condition][:,5] # R_vir
                halo_pop[:,7] = data[condition][:,20] # Mass
 
        elif (string == "all"):
            if extra_columns == True:
                halo_pop = np.zeros((np.shape(data)[0],8))
            else:
                halo_pop = np.zeros((np.shape(data)[0],6))
            for i in range(6):
                halo_pop[:,i] = data[:,8+i]
            if (extra_columns):
                halo_pop[:,6] = data[:,5] # R_vir
                halo_pop[:,7] = data[:,20] # Mass
        return halo_pop;

    
#     def alpha_list(self, vel, velocity_bulk):
#         """
#          Given two 3D arrays vel and velocity_bulk, this function computes the correlation coefficent of each vector in vel with the velocity_bulk vector. 
#     Returns the average, standard deviation, and number of vectors in vel that have been used for calculation.
    
#     Parameters:
#         vel (3D array): an array of 3D vectors representing the velocities of halos/particles
#         velocity_bulk (3D array): a 3D vector representing the bulk velocity of a given sub-box
        
#     Returns:
#         A list of correlation coefficents!
#         """
#         tmp = np.zeros((np.shape(vel)[0]))
#         for i in range(np.shape(vel)[0]):
#             tmp[i] = np.dot(vel[i], velocity_bulk)/(np.linalg.norm(vel[i]))/(np.linalg.norm(velocity_bulk))
#         # avg =  np.mean(tmp)
#         # std = np.std(tmp);
        
#         return tmp # returns sum of all dot product and number of halos in that cell

#     def sum_alpha(self, vel, velocity_bulk):
#         """
#          Given two 3D arrays vel and velocity_bulk, this function computes the correlation coefficent of each vector in vel with the velocity_bulk vector and returns all the computed alphas!
#     Returns the average, standard deviation, and number of vectors in vel that have been used for calculation.
    
#     Parameters:
#         vel (3D array): an array of 3D vectors representing the velocities of halos/particles
#         velocity_bulk (3D array): a 3D vector representing the bulk velocity of a given sub-box
        
#     Returns:
#         A list of the form [average, standard deviation, number of vectors in vel]
#         """
#         tmp = np.zeros((np.shape(vel)[0]))
#         for i in range(np.shape(vel)[0]):
#             tmp[i] = np.dot(vel[i], velocity_bulk)/(np.linalg.norm(vel[i]))/(np.linalg.norm(velocity_bulk))
#         avg =  np.mean(tmp)
#         std = np.std(tmp);
        
#         return [avg, std, np.shape(vel)[0]] # returns sum of all dot product and number of halos in that cell

    def sum_vel_dot_vbulk(self, vel, velocity_bulk):
        """
         Given two 3D arrays vel and velocity_bulk, this function computes the dot product of each vector in vel with the velocity_bulk vector. 
    Returns the average, standard deviation, and number of vectors in vel that have been used for calculation.
    
    Parameters:
        vel (3D array): an array of 3D vectors representing the velocities of halos/particles
        velocity_bulk (3D array): a 3D vector representing the bulk velocity of a given sub-box
        
    Returns:
        A list of the form [average, standard deviation, number of vectors in vel]
        """
        avg = np.mean(np.dot(vel, velocity_bulk))
        std = np.std(np.dot(vel, velocity_bulk))
        return [avg, std, np.shape(vel)[0]] # returns sum of all dot product and number of halos in that cell

    def vels_in_cell(self, ngrid, pos, vel, sub_box_index):
        boxsize = self.boxsize;
        """
        Given positions, velocities, and a sub-box index, this function returns the velocities of halos that are located in the specified sub-box.

            Parameters:
                ngrid (int): number of grids in the simulation
                pos (3D array): an array of 3D vectors representing the positions of halos
                vel (3D array): an array of 3D vectors representing the velocities of halos
                sub_box_index (list): a list of 3 integers representing the index of the sub-box (x, y, z)

            Returns:
                A 3D array containing the velocities of halos in the specified sub-box
        """
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        vel_halos =vel[condition_tot];
        # pos_halos =pos[condition_tot]# For test you can output pos[vel_halos, pos_halos];
        return vel_halos
    
    def pos_in_cell(self, ngrid, pos, vel, sub_box_index):
        boxsize = self.boxsize;
        """
        Given positions, velocities, and a sub-box index, this function returns the velocities of halos that are located in the specified sub-box.

            Parameters:
                ngrid (int): number of grids in the simulation
                pos (3D array): an array of 3D vectors representing the positions of halos
                vel (3D array): an array of 3D vectors representing the velocities of halos
                sub_box_index (list): a list of 3 integers representing the index of the sub-box (x, y, z)
            Returns:
                A 3D array containing the positions of halos in the specified sub-box
        """
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        # vel_halos =vel[condition_tot];
        pos_halos =pos[condition_tot]# For test you can output pos[vel_halos, pos_halos];
        return pos_halos

    def velocity_bulk(self, ngrid, pos,vel, sub_box_index):
        boxsize = self.boxsize;
        """
        Given positions, velocities, and a sub-box index, this function computes the bulk velocity and its standard deviation in the specified sub-box.
    
    Parameters:
        ngrid (int): number of grids in the simulation
        pos (3D array): an array of 3D vectors representing the positions of halos
        vel (3D array): an array of 3D vectors representing the velocities of halos
        sub_box_index (list): a list of 3 integers representing the index of the sub-box (x, y, z)
        
    Returns:
        A list of the form [[velocity_x, velocity_y, velocity_z], [standard deviation_x, standard deviation_y, standard deviation_z]]
        """
        condition_x = (np.floor(pos[:,0]*ngrid/boxsize)== sub_box_index[0])
        condition_y = (np.floor(pos[:,1]*ngrid/boxsize)== sub_box_index[1])
        condition_z = (np.floor(pos[:,2]*ngrid/boxsize)== sub_box_index[2])
        condition_tot = (condition_x) & (condition_y) & (condition_z)
        vel_grids= np.array([np.average(vel[condition_tot,0]),np.average(vel[condition_tot,1]),np.average(vel[condition_tot,2])])
        sigma_vel_grids= np.array([np.std(vel[condition_tot,0]),np.std(vel[condition_tot,1]),np.std(vel[condition_tot,2])])
        return [vel_grids,sigma_vel_grids];


    def catalogue_header(self, address, column=-1):
        """
        Given the address of a halo catalogue, this function reads the header and prints it. If a column index is specified, it will only print that specific column.
    
    Parameters:
        address (str): the filepath of the halo catalogue
        column (int): the index of the column to print, -1 to print the whole header (default)
        
    Returns:
        None
        """
        f = open(address)
        header = f.readline()
        header_list = header.split(' ') #list of columns
        if column==-1:
            for i in range(np.shape(header_list)[0]):
                print(str(i)+":"+header_list[i])
        if column!=-1:
            print(str(column)+":"+header_list[column])

    def gadget_load(self, snapshot, ptype=[1]):
        """
        Given the filepath of a gadget 2 snapshot and a list of particle types, this function returns the positions and velocities of the specified particle types.
    
    Parameters:
        snapshot (str): the filepath of the gadget 2 snapshot
        ptype (list): list of integers representing the particle types to extract, [1] for CDM and [2] for neutrinos (default)
        
    Returns:
        A 2D array of shape (2, N) containing the positions and velocities of the specified particle types, where N is the number of particles.
        """
        pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
        vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
        # ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
        return np.array([pos,vel])

    
    def Random_sub_box(self, ngrid, number):
        """
        Given the number of grids and the number of sub-boxes, this function generates and returns N randomly chosen sub-box indices.
    
    Parameters:
        ngrid (int): number of grids in the simulation
        number (int): number of randomly chosen sub-boxes
        
    Returns:
        A 2D array of shape (number, 3) containing the randomly chosen sub-box indices.
        """
        res = np.zeros((number,3)).astype(int)
        for i in range(number):
            for j in range(3):
                res[i,j] = random.randint(0,ngrid-1)      
        return res


    def data_sub_boxes(self, sub_box_lists, ngrid, pos_pcl, vel_pcl, pos_p1, vel_p1, pos_p2, vel_p2):
        """
    Computes alpha and sigma_alpha for each sub-box, and returns all alphas for the given number of sub-boxes.
    
    Parameters:
        sub_box_lists (list): index of sub-boxes to consider
        ngrid (int): Number of grids in the simulation, used to compute alpha over
        pos_pcl (np.array): N x 3 array containing the x, y, z positions of particles used to compute the bulk velocity
        vel_pcl (np.array): N x 3 array containing the vx, vy, vz velocities of particles used to compute the bulk velocity
        pos_p1 (np.array): N_p1 x 3 array containing the x, y, z positions of population 1
        vel_p1 (np.array): N_p1 x 3 array containing the vx, vy, vz velocities of population 1
        pos_p2 (np.array): N_p2 x 3 array containing the x, y, z positions of population 2
        vel_p2 (np.array): N_p2 x 3 array containing the vx, vy, vz velocities of population 2
        
    Returns:
        A list containing [index of sub-box, average of alpha for population 1, average of alpha for population 2, 
        standard deviation of alpha for population 1, standard deviation of alpha for population 2, 
        number of halos of population 1, number of halos of population 2] for each sub-box.
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
        Parallelized the computation of alpha and sigma_alpha for each sub-box using MPI.
        The function takes a list of sub-boxes, the number of grids, positions and velocities of 3 different particle populations.
        It distributes the sub-boxes among the available CPUs and computes the bulk velocity and velocities of population 1 and 2 in each sub-box.
        Then it computes alpha and sigma_alpha for population 1 and 2.
        It returns an array containing the sub-box index, average of alpha and sigma_alpha for population 1 and 2.
        
        Parameters:
    sub_box_lists (list): a list of indices of sub-boxes to compute alpha and sigma_alpha for
    ngrid (int): the number of grids
    pos_pcl (list or array): N x 3 array containing the positions of particles used to compute the bulk velocity
    vel_pcl (list or array): N x 3 array containing the velocities of particles used to compute the bulk velocity
    pos_p1 (list or array): N_p1 x 3 array containing the positions of population 1
    vel_p1 (list or array): N_p1 x 3 array containing the velocities of population 1
    pos_p2 (list or array): N_p1 x 3 array containing the positions of population 2
    vel_p2 (list or array): N_p1 x 3 array containing the velocities of population 2
    
Returns:
    An array containing [index of sub-box, average of alpha for population 1, average of alpha for population 2, std of alpha for population 1, std of alpha for population 2]
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
    