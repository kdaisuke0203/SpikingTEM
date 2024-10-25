import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

@tf.function
def generate_poisson_spikes(rates, T):
    """
    Generate spike times for neurons with constant firing rates using a Poisson process.
    
    Parameters:
    rates (ndarray): Array of firing rates (spikes per second) for each neuron.
    T (float): Total duration of the simulation (seconds).
    
    Returns:
    spike_times_list (list): List of lists of spike times for each neuron.
    """
    spike_times_list = []
    
    # 1つのニューロンごとにスパイクを生成
    for neuron_rates in rates:
        #print("neuron_rates", neuron_rates)
        for rate in neuron_rates:
            #print("rate", rate)
            spike_times = []
            t = 0
            #print("EE")
            #while t < 10:
                # Generate the next spike interval
            interval = -np.log(np.random.rand()) / rate
            t += interval
            #print("t",t)
            
            #if t < T:
            spike_times.append(t)
            
            spike_times_list.append(spike_times)
    
    return spike_times_list

# Example usage
"""rates = np.random.uniform(1, 10, size=(3, 5))  # Generate random rates for shape (16, 5)
T = 20  # total duration in seconds
spike_times_list = generate_poisson_spikes(rates, T)

# Plotting the spike trains
plt.figure(figsize=(12, 8))
for i, spike_times in enumerate(spike_times_list):
    plt.eventplot(spike_times, orientation='horizontal', colors='black', lineoffsets=i)

plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Poisson Spike Train for Multiple Neurons')
plt.show()"""
