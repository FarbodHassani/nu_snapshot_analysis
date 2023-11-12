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
import time

from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
M = 1.e14
z = 0.0
# nu = peaks.peakHeight(M, z)
# b = bias.haloBiasFromNu(nu, model = 'sheth01')
bias = bias.haloBias(M, model = 'tinker10', z = z, mdef = '200m')
print(bias)
from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');

do_the_analysis =True

Mass_cuts = [1.e12, 5.e12, 1.e13, 2.e13, 3.e13, 4.e13, 5.e13, 6.e13, 7.e13, 8.e13, 9.e13, 1.e14, 2.e14, 5.e14,7.e14, 1.e15];
ngrid_min = 1
ngrid_max = 160 # 160
ngrid_step = 2
ngrid_list = range(ngrid_min, ngrid_max, ngrid_step);
sims = ["0meV", "200meV", "300meV", "60meV"]

if do_the_analysis == True:
    for Mass_cut in Mass_cuts:
        df = pd.DataFrame()
        print("The analysis is being done for Mass_cut:"+f"{Mass_cut:.2e}")
        ## Halo model:
        model = 'sheth01'
        z = 0.0
        mdef = '200m'
        for sim in sims:
            for n_grid in ngrid_list:
                address_save = "./Analysis_sims_L_2048_Ngrid_4096//"+sim;
                file_path = address_save+"/ngrid_"+str(n_grid)+".npy"
                if os.path.exists(file_path):
                    a = np.load(file_path, allow_pickle=True)
                    ######
                    mass_conditions = (a[:,6]>=Mass_cut)
                    a = a[mass_conditions]
                    masses = a[:,6];                    
                    #######
                    v_bulk_x = a[:,3];
                    v_bulk_y = a[:,4];
                    v_bulk_z = a[:,5];
                    biases = bias.haloBias(masses, model = 'sheth01', z = 0.0, mdef = '200m')
                    x1 = (biases + (masses/(1.3*1.e14))**(0.85))*v_bulk_x;
                    x2 = (biases + (masses/(1.3*1.e14))**(0.85))*v_bulk_y;
                    x3 = (biases + (masses/(1.3*1.e14))**(0.85))*v_bulk_z; 
                    y1 = a[:,0]; # vi_x - v_b_x
                    y2 = a[:,1]; # vi_x - v_b_x
                    y3 = a[:,2]; # vi_x - v_b_x

                    cor_x = np.average(x1*y1)
                    cor_y = np.average(x2*y2)
                    cor_z = np.average(x3*y3)
                    cor_all_vel = (cor_x + cor_y + cor_z)/(np.average(v_bulk_x*v_bulk_x) + np.average(v_bulk_y*v_bulk_y) + np.average(v_bulk_z*v_bulk_z));
                    cor_all = (cor_x + cor_y + cor_z)/(np.average(x1*x1) + np.average(x2*x2) + np.average(x3*x3));

                    # create a dictionary with the values
                    row_dict = {'sim': sim, 'n_grid': n_grid, 'dx': 1024./n_grid,  'mass_cut': Mass_cut, 'cor_x': cor_x, 'cor_y': cor_y, 'cor_z': cor_z, 'cor_all': cor_all, 'cor_all_vel': cor_all_vel}
                    # append the row to the dataframe
                    df = df.append(row_dict, ignore_index=True)
                else:
                    print(file_path+" doesn't exist!")
        df.to_csv('./correlation_data_L_2048_N_4098/data_L_2048_N_4096_mass_cut_'+f"{Mass_cut:.2e}"'.csv', index=False);

else:
    df=[]
    for Mass_cut in Mass_cuts:
        print("The file "+f"{Mass_cut:.2e}"+" are being read!")
        df.append(pd.read_csv('./correlation_data_L_2048_N_4098/data_L_2048_N_4096_mass_cut_'+f"{Mass_cut:.2e}"'.csv'))

