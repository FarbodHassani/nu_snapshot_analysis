import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_sub_box(positions, velocities, positions2, velocities2, average_velocity, average_velocity2, ps):
    

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the positions of the particles as points
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], s=ps, color="red", alpha=0.4, label="type 1")
    ax.scatter(positions2[:,0], positions2[:,1], positions2[:,2], s=ps, color="blue", alpha=0.4, label="type 2")

    # Plot the velocity vectors for each particle
    for i in range(len(positions)):
        ax.quiver(positions[i,0], positions[i,1], positions[i,2], velocities[i,0], velocities[i,1], velocities[i,2], color='r')
    for i in range(len(positions2)):
        ax.quiver(positions2[i,0], positions2[i,1], positions2[i,2], velocities2[i,0], velocities2[i,1], velocities2[i,2], color='b')

    # Plot the average velocity vector
    ax.quiver(positions[:,0].mean(), positions[:,1].mean(), positions[:,2].mean(), average_velocity[0], average_velocity[1], average_velocity[2], color='k', label= " bulk velocity cdm")
    ax.quiver(positions[:,0].mean(), positions[:,1].mean(), positions[:,2].mean(), average_velocity2[0], average_velocity2[1], average_velocity2[2], color='green',label= " bulk velocity halos")
    
    plt.legend(fontsize =20)
    plt.show()
    
    
    
def smooth_data(x, y, x_new):
    # Compute the bin edges for the new bins
    bin_edges = np.histogram_bin_edges(x_new, bins=x_new)
    
    # Compute the bin indices for the original data
    bin_indices = np.digitize(x, bin_edges)    
    # Compute the bin averages and standard deviations
    bin_averages = np.zeros_like(bin_edges[:-1])
    bin_stddevs = np.zeros_like(bin_edges[:-1])
    for i in range(len(bin_edges) - 1):
        bin_data = y[bin_indices == i+1]
        if len(bin_data) == 0:
            print(f"Warning: No data in bin {i+1}")
            bin_averages[i] = np.nan
            bin_stddevs[i] = np.nan
        else:
            bin_averages[i] = np.mean(bin_data)
            bin_stddevs[i] = np.std(bin_data, ddof=1)
    
    # Return the new bin edges, averages, and standard deviations
    return bin_edges[:-1], bin_averages, bin_stddevs