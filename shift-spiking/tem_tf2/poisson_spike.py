import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_poisson_spikes(rates, T=10):
    """
    Generate spike times for neurons with constant firing rates using a Poisson process.
    
    Parameters:
    rates (ndarray): Array of firing rates (spikes per second) for each neuron.
    T (float): Total duration of the simulation (seconds).
    
    Returns:
    spike_times_list (list): List of lists of spike times for each neuron.
    """
    spike_times_list = []
    bin_num = 20
    dt_min = 1 / bin_num
    cell_num = rates.shape[1]
    env_num = rates.shape[0]
    #print("ce",cell_num, env_num)
    spike_train = np.zeros((env_num, cell_num, bin_num))
    for i in range(env_num):
        print("i", i)
        for j in range(cell_num):
            print("j", j)
            spike_times = []
            t = 0
            while t < T:
                # Generate the next spike interval
                interval = -np.log(np.random.rand()) / rates[i][j] + dt_min
                t += interval 
                t = np.round(t,2)
                
                if t < T:
                    spike_times.append(t)
                    spike_train[i][j][int(t*bin_num)] = 1.0 
            spike_times_list.append(spike_times)     

    #print("d",spike_times_list)
    #print("spi",spike_train[0])
    
    return spike_times_list

# Example usage
"""rates = np.random.uniform(1, 10, size=(3, 5))  # Generate random rates for shape (16, 5)
T = 30  # total duration in seconds
spike_times_list = generate_poisson_spikes(rates, T)

# Plotting the spike trains
plt.figure(figsize=(12, 8))
for i, spike_times in enumerate(spike_times_list):
    plt.eventplot(spike_times, orientation='horizontal', colors='black', lineoffsets=i)

plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Poisson Spike Train for Multiple Neurons')
plt.show()"""
