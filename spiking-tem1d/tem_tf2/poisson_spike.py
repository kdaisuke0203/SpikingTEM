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
    cell_num = rates.shape[1]
    env_num = rates.shape[0]
    #print("ce",rates)
    spike_train = np.zeros((env_num, cell_num, T))
    for i in range(env_num):
        #print("i", i)
        for j in range(cell_num):
            #print("j", j)
            t = 0
            while t < T:
                # Generate the next spike interval
                interval = -np.log(np.random.rand()) / (rates[i][j] + 1e-5)
                t += interval 
                t = np.round(t,0)
                #print("ttttt",t,T)
                if t < T:
                    spike_train[i][j][int(t)] = 1.0   
    
    return spike_train

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
