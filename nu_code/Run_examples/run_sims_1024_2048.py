# to run: mpiexec -n 4 python run_sims.py


import sys
sys.path.append('/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/nu_code/')
# from library_snapshot import bulk_velocity # For the x-y analysis
# from bulk_velocity import bulk_calculation
from library_snapshot import computation_xy, analysis_functions
from computation_xy import compute_regression
from analysis_functions import perform_analysis

#################
## Global variables
#################
sims= ["0.0ev", "0.15ev", "0.3ev", "0.6ev", "0.9ev"];
specs = ["L_1024_Ngrid_2048"]# specs = ["L_1024_Ngrid_512"]#, "L_1024_Ngrid_1024"]
Mass_cuts =  [1.e13, 5.e13, 7.e13, 1.e14, 2.e14];
boxsize = 1024.
ngrid_max = 80
halo_dir = "/mn/stornext/u3/hassanif/neutrino_niayesh/simulations/"
save_path = "./correlations_xy_"+specs[0]
ngrid_step = 1
ngrid_list = range(1, ngrid_max, ngrid_step)
num_cores = 25
cdm_analysis = True
nu_analysis = True
iteration_num = 5
n_h_threshold = 3
coeff_halo = 1.

perform_analysis(sims, specs, Mass_cuts, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, coeff_halo, num_cores, cdm_analysis, nu_analysis, n_h_threshold)

for spec in specs:
    for sim in sims:
        compute_regression(Mass_cuts, spec, sim, boxsize, ngrid_max, halo_dir, save_path, ngrid_step, ngrid_list, iteration_num, n_h_threshold, run_tests=False, remove_extra_files=False)