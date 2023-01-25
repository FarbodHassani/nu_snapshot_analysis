import os
import psutil
from collections import defaultdict


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


def memory():
    psutil.virtual_memory()
    # you can convert that object to a dictionary 
    dict(psutil.virtual_memory()._asdict())
    pid = os.getpid()
    python_process = psutil.Process(pid)
    # you can have the percentage of used RAM
    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse, 'GB, percentage: ',psutil.virtual_memory().percent)
    
 
