#!/bin/bash

### To run the code: bash Run.sh 0 1 --> means to run cases of 0.0ev and 0.15ev when having loop over cdm, nu and halos
# Check if arguments are provided, otherwise, set to empty
if [ -z "$1" ]; then
    args=()
else
    args=("$@")
fi

names=("0.0ev" "0.2ev" "0.3ev" "0.06ev")
#names2=("nu" "cdm" "halo_mass_5e13")
names2=("halo_mass_1e12" "halo_mass_2e12" "halo_mass_3e12" "halo_mass_4e12" "halo_mass_5e12" "halo_mass_6e12" "halo_mass_7e12" "halo_mass_8e12" "halo_mass_9e12" "halo_mass_1e13" "halo_mass_2e13" "halo_mass_3e13" "halo_mass_4e13" "halo_mass_5e13" "halo_mass_6e13" "halo_mass_7e13" "halo_mass_8e13" "halo_mass_9e13" "halo_mass_1e14" "halo_mass_2e14")

simulations_dir="/mn/stornext/u3/hassanif/neutrino_niayesh/simulations/L_2048_Ngrid_4096/"

# Check if arguments are within the valid range for names array
valid_args=()
for arg in "${args[@]}"; do
    if ((arg >= 0 && arg < ${#names[@]})); then
        valid_args+=("${names[$arg]}")
    fi
done

# If no valid arguments provided, use all names in the names array
if [ ${#valid_args[@]} -eq 0 ]; then
    valid_args=("${names[@]}")
fi

# Get the hostname
hostname=$(hostname)
org_dir="/mn/stornext/u3/hassanif/neutrino_niayesh/Analysis/Runs_bulk_all_sims/L_2048_Ngrid_4096"

# Create a log file with the hostname to store warnings and errors
# log_file="script_log_${hostname}.txt"
# > "$log_file"  # Clear the log file if it already exists


for name in "${valid_args[@]}"
do
    for name2 in "${names2[@]}"
    do
        # Redirect both stderr and stdout to the log file
        {
        # Create directory with the same name as the current loop name
        rm -r "$name/$name2"
        mkdir -p "$name/$name2"
        echo "$name/$name2"
        if [[ $name2 == "nu" ]]; then
            sed -e "s#file_path = file#file_path = ${simulations_dir}${name}/output/snap002_ncdm0#g" -e "s#sim_type = xev#sim_type = ${name}#g" -e "s#bulk_species = species#bulk_species = nu#g" settings.ini > ./"$name"/"$name2"/settings.ini
        elif [[ $name2 == "cdm" ]]; then
            sed -e "s#file_path = file#file_path = ${simulations_dir}${name}/output/snap002_cdm#g" -e "s#sim_type = xev#sim_type = ${name}#g" -e "s#bulk_species = species#bulk_species = cdm#g" settings.ini > ./"$name"/"$name2"/settings.ini

        # elif [[ $name2 == "halo_mass_"* ]]; then
        #     sed -e "s#file_path = file#file_path = ${simulations_dir}${name}/output/halos/out_2.list#g" -e "s#bulk_species = species#bulk_species = halo#g" settings.ini > ./"$name"/"$name2"/settings.ini
        elif [[ $name2 == "halo_mass_"* ]]; then
            mass_limit="${name2#halo_mass_}"  # Extract the mass limit from name2
            sed -e "s#file_path = file#file_path = ${simulations_dir}${name}/nu_${name}_2048box_4096_snap004_halos_parents.dat#g" -e "s#bulk_species = species#bulk_species = halo#g" -e "s#sim_type = xev#sim_type = ${name}#g" -e "s#mass_limit = 1.e12#mass_limit = ${mass_limit}#g" settings.ini > ./"$name"/"$name2"/settings.ini
        fi

       # Copy necessary files to directory
        cp main.py run.sh ./"$name"/"$name2"
        # mv ./"$name"/settings_$name.ini ./"$name"/settings.ini
        (cd "$org_dir"/"$name"/"$name2" && nohup bash run.sh > nohup.out 2>&1 &)
        cp main.py run.sh ./"$name"/"$name2"

        
        } 

    done
done
# Print the log file content at the end
# cat "$log_file"
