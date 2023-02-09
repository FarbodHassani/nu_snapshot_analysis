import re

def parse_parameter_file(file_path):
    """
    Parses a parameter file and returns a dictionary of the parameters and their values.
    Assumes that the parameter file is in the format "parameter = value" (without quotes).
    """
    # Initialize an empty dictionary to store the parameters and their values
    parameters = {}

    # Open the parameter file
    with open(file_path, 'r') as f:
        # Read through each line of the file
        for line in f:
            # Use a regular expression to match the parameter and its value
            match = re.match(r'(\w+) = (.+)', line)

            # If a match is found
            if match:
                # Extract the parameter and value from the match object
                parameter = match.group(1)
                value = match.group(2)

                # Strip any leading or trailing whitespace from the value
                value = value.strip()
                # Handle the case where the parameter is 'sim_type'
                if parameter == 'sim_type':
                    if ',' in value:
                        # Split the value on ',' and store as a list
                        value = value.split(',')
                        value = [x.strip() for x in value]
                    else:
                        value = [value]
                elif parameter == "simulation":
                    value = [value]        
                
                elif parameter == "number_samples":
                    value = list(map(int, value.strip("[]").split(",")))
                    
                elif parameter in ["boxsize", "mass_limit"]:
                    value = float(value)
                    
                elif parameter == "sim_path":
                    value = value
                    
                elif parameter == "remove_subhalos":
                    value = value     
                    
                elif parameter == "snapshot_number":
                    value = [int(value)]
                    
                    
                elif parameter in ["ngrid_min", "ngrid_max", "ngrid_step"]:
                    value = int(value)
                    
                elif parameter == "save_path":
                    value = value
                    
                else:
                    value = [value]  

                 # Store the parameter and value in the dictionary
                parameters[parameter] = value
                print(f'{parameter}: {value}')
                
    return parameters


def string_to_int_list(string_list):
    return list(map(int, string_list.strip("[]").split(",")))