import os
import re
import pandas as pd
import pickle
import numpy as np

def process_directory(input_dir, output_dir):
    """
    This function takes in two inputs, the input_dir and output_dir. It goes through the files in the input_dir directory,
    reads their contents, performs some computation on the contents, and saves the results to the output_dir directory.

    Parameters:
    - input_dir (str): The directory to search for files.
    - output_dir (str): The directory to save the results to.
    """
    column_names = ['simulation','sim_info', 'snapshot_num','ngrid', 'dx','number_samples', 'alpha_p1', 'alpha_p2', 'std_alpha_p1', 'std_alpha_p2', 'num_subbox_analysed']
    data_alpha = pd.DataFrame(columns=column_names)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            with open(file_path,'rb') as handle:
                
                L_match = re.search(r"L_(\d+)", file)
                boxsize = int(L_match.group(1))
                
                N_match = re.search(r"N_(\d+)", file)
                N_grid_sim = int(N_match.group(1))
                
                snap_match = re.search(r"snap_(\d+)", file)
                snap_num = int(snap_match.group(1))
                
                ngrid_match = re.search(r"_ngrid_(\d+)", file)
                ngrid = int(ngrid_match.group(1))

                number_samples_match = re.search(r"Nsubboxes_(\d+)", file)
                number_samples = int(number_samples_match.group(1))
                
                df = pickle.load(handle)
                if not df['bulk_vel_cdm_i'].empty:
                    vb = np.array([np.array(x) for x in df['bulk_vel_cdm_i']])[0]# The zeroth component picks the velocities the first component is the error!
                    vb_dot_vb = np.einsum('ij,ij->i', vb, vb) # dot product of v_bulk vectors with itself
                    avg_vb_dot_vb = np.average(vb_dot_vb) # since there are many particles the weights for vb_dot_vb are more or less close to 1
                    std_vb_dot_vb = np.sqrt(np.average((vb_dot_vb - np.average(vb_dot_vb))**2))
                    # print()
                    #### population 1
                    weight = np.array([np.array(x) for x in df['vp1.v_b(halos)']])[:,2] # number of halos in each sub-box
                    vector_vp1_dot_vb = np.array([np.array(x) for x in df['vp1.v_b(halos)']])[:,0]
                    avg_vp1_dot_vb = np.average(vector_vp1_dot_vb, weights=weight, axis=0)
                    std_vp1_dot_vb=  np.average((vector_vp1_dot_vb - np.average(vector_vp1_dot_vb, weights=weight))**2, weights=weight) # This computes the standard deviation considering the weights!

                    #### population 2
                    weight = np.array([np.array(x) for x in df['vp2.v_b(halos)']])[:,2] # number of halos in each sub-box
                    vector_vp2_dot_vb = np.array([np.array(x) for x in df['vp2.v_b(halos)']])[:,0]
                    avg_vp2_dot_vb = np.average(vector_vp2_dot_vb, weights=weight, axis=0)
                    std_vp2_dot_vb=  np.average((vector_vp2_dot_vb - np.average(vector_vp2_dot_vb, weights=weight))**2, weights=weight) # This computes the standard deviation considering the weights!

                    # computing alpha_p1, alpha_p2
                    # std_alpha_p1 = (mu/nu)^2 * (std_mu^2/mu^2 + std_nu^2/nu^2) he ratio of two averages, alpha = mu/nu, can be thought of as the result of dividing two normal distributions. The variance of the ratio of two random variables is given by:
                    # Var(alpha) = (mu/nu)^2 * (Var(mu)/mu^2 + Var(nu)/nu^2 - 2 * Cov(mu,nu)/(mu*nu))
                    alpha_p1 = avg_vp1_dot_vb/avg_vb_dot_vb
                    std_alpha_p1 = (avg_vp1_dot_vb/avg_vb_dot_vb)**2 * (std_vp1_dot_vb**2/avg_vp1_dot_vb**2 + std_vb_dot_vb**2/avg_vb_dot_vb**2)
                    # std_alpha_p1 = np.sqrt(var_alpha_p1); 

                    # pop2:
                    alpha_p2 = avg_vp2_dot_vb/avg_vb_dot_vb
                    var_alpha_p2 = (avg_vp2_dot_vb/avg_vb_dot_vb)**2 * (std_vp2_dot_vb**2/avg_vp2_dot_vb**2 + std_vb_dot_vb**2/avg_vb_dot_vb**2)
                    std_alpha_p2 = np.sqrt(var_alpha_p2); 
                    ### saving data:
                    dx = boxsize/ngrid;
                    sim = np.array([np.array(x) for x in df['simulation']])[0];
                    sim_info = "L_"+str(boxsize)+"_N_"+str(N_grid_sim);
                    data_saved = {
                            'simulation':sim,
                            'sim_info':sim_info,
                            'snapshot_num':snap_num,
                            'ngrid':ngrid,
                            'dx':dx,
                            'number_samples':number_samples,
                            'alpha_p1':alpha_p1,
                            'alpha_p2':alpha_p2,
                            'std_alpha_p1':std_alpha_p1,
                            'std_alpha_p2':std_alpha_p2,
                            'num_subbox_analysed':np.shape(vb)[0]
                            }
                    data_alpha = data_alpha.append(data_saved, ignore_index=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+'/simulation_'+str(sim)+'_'+str(sim_info)+'_snap_'+str(snap_num)+'_statistics.pickle', 'wb') as handle:
        pickle.dump(data_alpha, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        

def load_pickle(name):
    """
    This function takes in a name as an input, and loads a pickle file with that name.

    Parameters:
    - name (str): The name of the pickle file to be loaded.

    Returns:
    - df (DataFrame): DataFrame that was loaded from pickle file.
    """
    try:
        # Open the pickle file with the provided name
        with open(name, 'rb') as handle:
            # Load the contents of the file into a DataFrame
            df = pickle.load(handle)
        return df
    except FileNotFoundError:
        print(f"{name} not found.")
    except:
        print(f"An error occurred while loading {name}.")