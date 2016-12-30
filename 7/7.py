#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import seaborn as sns

sns.set_style('dark')

"""
Policy iteration and Q-learning practice
"""

# TODO more general name
def plot_states(states, values, start=None, ends=set(), policy=dict()):
    plt.figure()
    # TODO cover range of state coordinates
    plt.axis([0,4,0,3])
    plt.grid(False)

    plt.title('States and their values, overlayed arrows represent policy')

    ax = plt.gca()
    ax.invert_yaxis()

    scale = 1
    width = scale
    height = width

    for s in states:
        # TODO black for missing states?
        # TODO use .text to overlay value
        if values[s] == 0:
            c="white"
        elif values[s] > 0:
            c="green"
        else:
            c="red"
        # for Rectangle: (x,y), width, height
        ax.add_patch(Rectangle((s[1] * scale, s[0] * scale), \
                width, height, facecolor=c))

        if s == start:
            plt.text(s[1] + width * 0.05, s[0] + height * 0.13, 'Start')
        elif s in ends:
            plt.text(s[1] + width * 0.05, s[0] + height * 0.13, 'End')

        # TODO how to plot bold?
        # how to draw arrows?
        if not values[s] == 0:
            plt.text(s[1] + width * 0.46, s[0] + height * 0.5, str(values[s]))

    # policies in our example map from states to neighboring states
    # more generally, they map to actions, which may not deterministically
    # bring you to the next state
    # TODO put values in bottom right for arrows in middle
    arrow_scale = 0.3
    for s1, s2 in policy.items():
        dy = (s2[1] - s1[1]) * arrow_scale
        dx = (s2[0] - s1[0]) * arrow_scale
        plt.arrow(s1[1] + width * 0.5, s1[0] + height * 0.5, dy, dx)

    plt.show()


# TODO cache these?
def get_neighbors(s, states):
    """ Returns a set of states s' that are neighbors of s in states """

    i, j = s
    ns = set()

    # TODO could definitely improve on this efficiency if statespace
    # ever gets large
    for s_prime in states:
        i_prime, j_prime = s_prime

        # we don't want it to be the same square
        # or anything further than one taxicab distance away
        if abs(i - i_prime) + abs(j - j_prime) == 1:
            ns.add(s_prime)

    return ns

def initial_policy(neighbors):
    """ Generate a random initial policy uniformly """

    policy = dict()

    for s, ns in neighbors.items():
        # pick a random next state
        # so our initial policy is defined everywhere
        # needs to be a neighbor of s
        policy[s] = random.choice(list(ns))

    return policy

# in general definition, is also conditional on the action a
def P(s, s_prime):
    # TODO what other variables? a and pi?
    return 1

def R(s, s_prime):
    """ Defines the rewards after making transitions from s to s_prime. """
    # TODO what other variables? a and pi?

    # green square gets +5
    if s_prime == (2,3):
        return 5
    # red square gets -5
    elif s_prime == (1,2):
        return -5
    else:
        return 0

def V(s, neighbors, V, discount):
    """ Calculate the expected future value of a single state s.
        Uses existing value estimate in calculation.  """

    acc = 0

    # calculate the expected value of a certain state
    # 'over a potentially infinite horizon'
    for s_prime in neighbors[s]:
        acc += P(s,s_prime) * (R(s,s_prime) + discount * V(s_prime))
    
    return acc

def update_policy(V_curr, states, neighbors):
    # doesn't actually need current policy (that just factors in to V)
    
    policy = dict()

    # we don't actually have to iterate over s_prime (as in wiki formula)
    # because there is no probability of landing in states not selected
    # by the action (in our problem)

    for s in states:
        # TODO could make negative infinity or equivalent
        # and policy[s] = whatever
        ns = list(neighbors[s])
        best = V_curr[ns[0]]
        policy[s] = ns[0]

        # but we do have to loop over possible actions, which are one-to-one with s_prime
        # (argmax loop in wiki formula)
        for i, s_prime in enumerate(neighbors[s]):
            if V_curr[neighbors[s][i]] > best:
                best = V_curr[ns[i]]
                policy[s] = ns[i]

    return policy

# TODO handle probability in way consistent w/ above
def update_values(V_prev, states, neighbors, discount):
    V_next = np.empty_like(V_prev) * np.nan

    for s in range(V_next.shape[0]):
        # TODO is this the right step 2?
        V_next[s] = V(states[s], neighbors, V_prev, discount)

    return V_next

# the figure homework 7
states = {(0,0),(0,1),(0,2),(0,3),\
          (1,0),      (1,2),(1,3),\
          (2,0),(2,1),(2,2),(2,3)}

# default is zero, constructed this way
# WARNING: if you try to read any missing states,
# the dictionary may enter them mapped to the default value
# (so that it gets iterated over and 'in' checks pass)
values = defaultdict(int)
values[(1,2)] = -5
values[(2,3)] =  5

start_state = (0,0)
# TODO values in this, or more generally somewhere?
end_states = {(1,2),(2,3)}

neighbors = dict()
for s in states:
    neighbors[s] = get_neighbors(s, states)

discount = 0.99

pi = initial_policy(neighbors)
# TODO values to initial random states too?
# a problem to do all zeros or whatever if some initial values are less than 0?
V_curr = np.zeros(len(pi))

'''
Implement a dynamic programming approach to solve this problem using policy it-
eration. Start with a random initial policy. Use a discount factor of γ = 0.99. For
each cell on the grid, display the value of the optimal value function V ∗ and show the
optimal policy π ∗ using arrows
'''

#plot_states(states, values, start_state, end_states, pi)

policy_iterations = 100
value_iterations = 1000

p = 0

# a set number of iterations, or until convergence of pi
while p < policy_iterations:
    # TODO not quite sure how to formulate this in terms of DP
    # so will first just try and implement equations in wiki on MDPs
    last_pi = pi
    pi = update_policy(V_curr, states, neighbors)
    # check for policy convergence
    if pi == last_pi:
        print('Policy converged.')
        break

    v = 0
    last_V = np.empty_like(V_curr) * np.nan

    # either for a certain number of iterations or until convergence
    while v < value_iterations and not np.allclose(V_prev, V_curr):
        last_V = V_curr
        # TODO no mutability issue right?
        V_curr = update_values(V_curr, states, neighbors, discount)
        v += 1

    p += 1
