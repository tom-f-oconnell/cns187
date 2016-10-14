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

V_passive = np.zeros(bins)

for t in range(1, bins):
    V_passive[t] = V_passive[t-1] + (I[t] - V_passive[t-1] / R_mem) / C_mem
    if not np.isnan(V_passive[t-1]) and np.isnan(V_passive[t]):
        print(I[t])
        print(V_passive[t-1] / R_mem)
        print((I[t] - V_passive[t-1] / R_mem) / C_mem)

V_lif = np.zeros(bins)
v_max = 10
v_thresh = 1

# TODO refractoriness
for t in range(1, bins):

    # reset if we just spiked
    if V_lif[t-1] == v_max:
        V_lif[t] = 0
    else:
        V_lif[t] = V_lif[t-1] + (I[t] - V_lif[t-1] / R_mem) / C_mem

        # spike if we reached threshold
        if V_lif[t] >= v_thresh:
            V_lif[t] = v_max

