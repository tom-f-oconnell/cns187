#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random

# Question 2: Integrate and Fire Neurons

# TODO test
def poisson_spikes(freq, seconds, dt):
    """ Returns a train of spikes sampled from a Poisson at mean frequency freq (Hertz) """
    
    bins = round(seconds / dt)
    train = np.zeros(bins)

    # expected number of spikes over the train
    E = freq * seconds
    # probability in one bin?
    p = E / bins

    for i in range(0, len(train)):
        if random.random() <= p:
            train[i] = 1
        else:
            train[i] = 0

    return train

seconds = 3
# refractory period should be around 2 ms
delta_t = 0.001
bins = round(seconds / delta_t)

# this cell is said to have a time constant of 100 ms
# since that arises from the resistance and capacitance (nothing else, right?)
# it seems strange to just manually set the time constant and vary these separately

R_mem = 1e8 # 100 MOhms

# assuming a spherical neuron
C_per_area = 1e-6 # 1 uF / cm^2
r = 25e-4         # 25 um radius (expressed in cm)
A = 4 * np.pi * r**2

C_mem = A * C_per_area  # Farads
# tau = R * C

c1 = poisson_spikes(5, seconds, delta_t)
c2 = poisson_spikes(10, seconds, delta_t)
c3 = poisson_spikes(30, seconds, delta_t)

# time constant
tau = 0.1 # seconds. 100 ms.

# 5 just to get nearly all of the effect
# could probably use less
kernel_bins = round(5 * tau / delta_t)
decay_kernel = np.empty(kernel_bins) * np.nan

"""
lam = 1 / tau
decay_kernel[0] = 1
for t in range(1, kernel_bins):
    decay_kernel[t] = decay_kernel[t-1] - lam * decay_kernel[t-1]
"""

I_0 = 1e-9 # 1 nA
for t in range(0, kernel_bins):
    decay_kernel[t] = I_0 * np.e**(-1 * t * delta_t / tau)

# need to normalize the decay kernel to not have the convolution amplify anything

I_1 = np.convolve(c1, decay_kernel, mode='same')
I_2 = np.convolve(c2, decay_kernel, mode='same')
I_3 = np.convolve(c3, decay_kernel, mode='same')

# TODO prevent coincidence first?
I = I_1 + I_2 + I_3

# problem with this approach is the R and C don't actually define the time constant
# (it seems) since they want us to adjust those separately
# TODO need current to reset though
# I = (c1 + c2 + c3) * I_0

V_passive = np.zeros(bins)

for t in range(1, bins):
    V_passive[t] = V_passive[t-1] + delta_t * (I[t] - V_passive[t-1] / R_mem) / C_mem

in_spikes = c1 + c2 + c3
I_spiking = np.zeros(bins)

V_lif = np.zeros(bins)
v_max = 1
v_thresh = 0.6

refractory_period = 2e-3 # 2 ms
refractory_bins = round(refractory_period / delta_t)

Cs = [0.01*C_mem, 0.1*C_mem, C_mem, 10*C_mem]

for C in Cs:
    for t in range(1, bins):

        # reset if we just spiked
        if V_lif[t-1] == v_max:
            V_lif[t] = 0

            I_spiking[t:] = np.zeros(len(I_spiking[t:]))
            in_spikes[t:t + refractory_bins] = np.zeros(refractory_bins)
        else:
            if in_spikes[t] >= 1:
                prev = I_spiking[t:t+len(decay_kernel)]
                I_spiking[t:t+len(decay_kernel)] = prev + decay_kernel[:len(prev)]
                
            V_lif[t] = V_lif[t-1] + delta_t * (I_spiking[t] - V_lif[t-1] / R_mem) / C

            # spike if we reached threshold
            if V_lif[t] >= v_thresh:
                V_lif[t] = v_max
    
    plt.figure()
    #plt.plot(V_lif)
    spikes = V_lif.copy()
    spikes[spikes < v_max] = 0
    # plt.matshow(np.vstack((c1,c2,c3,spikes)))
    plt.plot(spikes)
    plt.title('Membrane capacitance='+str(C)+' Farads, resistance='+str(R_mem)+' Ohms')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spike occurence for each cell')

Rs = [0.01*R_mem, 0.1*R_mem, R_mem, 10*R_mem]

for R in Rs:
    for t in range(1, bins):

        # reset if we just spiked
        if V_lif[t-1] == v_max:
            V_lif[t] = 0

            I_spiking[t:] = np.zeros(len(I_spiking[t:]))
            in_spikes[t:t + refractory_bins] = np.zeros(refractory_bins)
        else:
            if in_spikes[t] >= 1:
                prev = I_spiking[t:t+len(decay_kernel)]
                I_spiking[t:t+len(decay_kernel)] = prev + decay_kernel[:len(prev)]
                
            V_lif[t] = V_lif[t-1] + delta_t * (I_spiking[t] - V_lif[t-1] / R) / C_mem

            # spike if we reached threshold
            if V_lif[t] >= v_thresh:
                V_lif[t] = v_max
    
    plt.figure()
    #plt.plot(V_lif)
    spikes = V_lif.copy()
    spikes[spikes < v_max] = 0
    #plt.matshow(np.vstack((c1,c2,c3,spikes)))
    plt.plot(spikes)
    plt.title('Membrane capacitance='+str(C_mem)+' Farads, resistance='+str(R)+' Ohms')
    plt.xlabel('Time (seconds)')
    #plt.ylabel('Volts')
    plt.ylabel('Spike occurence for each cell')
