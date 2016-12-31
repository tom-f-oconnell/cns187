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
def plot_states(states, values, start=None, ends=set(), policy=dict(), discount=0.99):
    # display terms with values plugged in, but unevaluated
    # on the grid, for debugging
    plot_equations = True

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
        if not values[s] == 0:
            plt.text(s[1] + width * 0.65, s[0] + height * 0.94, '%.1f' % values[s])

    # policies in our example map from states to neighboring states
    # more generally, they map to actions, which may not deterministically
    # bring you to the next state
    # TODO put values in bottom right for arrows in middle
    arrow_scale = 0.3
    for s1, s2 in policy.items():
        dy = (s2[1] - s1[1]) * arrow_scale
        dx = (s2[0] - s1[0]) * arrow_scale
        plt.arrow(s1[1] + width * 0.5, s1[0] + height * 0.5, dy, dx)

    if plot_equations:
        for s in states:
            # TODO actually pass discount
            plt.text(s[1] + width * 0.1, s[0] + height * 0.6, \
                '%.1f * (%.1f + \n%.2f * %.1f)' % \
                (1.0, R(s, policy[s]), discount, V(s, policy, values, discount)))

        plt.xlabel('P(s, s_prime) * (R(s, s_prime) + discount * values[s_prime])')


    plt.show()

def values_equal(values1, values2):
    """ Compare two dicts that have float values, using np.isclose to compare them. """

    # they must be dicts
    if not type(values1) == dict:
        return False
    if not type(values2) == dict:
        return False

    for k, v in values1.items():
        # TODO check that this function returns False (under expected operation)
        # because of the np.close statement
        if not (k in values2 and np.isclose(values1[k], values2[k])):
            return False

    # make sure values2 doesn't have extra elements
    if len(values2) == len(values1):
        return True

    return False

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

def initial_policy(neighbors, ends=set()):
    """ Generate a random initial policy uniformly """

    policy = dict()

    for s, ns in neighbors.items():
        if s in ends:
            # TODO maybe handle some other way?
            # if this won't by itself allow convergence
            # end states should not allow outward transitions (?)
            policy[s] = s
        else:
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

    if s == s_prime:
        return 0

    # green square gets +5
    if s_prime == (2,3):
        return 5
    # red square gets -5
    elif s_prime == (1,2):
        return -5
    else:
        return 0

def V(s, policy, values, discount):
    """ Calculate the expected future value of a single state s.
        Uses existing value estimate in calculation.  """

    '''
    acc = 0

    # calculate the expected value of a certain state
    # 'over a potentially infinite horizon'
    for s_prime in neighbors[s]:
        # TODO correct w/ values and everything? P evaluation might be wrong...
        acc += P(s,s_prime) * (R(s,s_prime) + discount * values[s_prime])

    return acc
    '''
    # generally we might need the neighbors, but for this problem
    # the policy is fine (deterministic)

    # TODO implement more generally applicable version

    s_prime = policy[s]
    return P(s, s_prime) * (R(s, s_prime) + discount * values[s_prime])

def update_policy(V_curr, states, neighbors, ends=set()):
    # doesn't actually need current policy (that just factors in to V)
    
    policy = dict()

    # we don't actually have to iterate over s_prime (as in wiki formula)
    # because there is no probability of landing in states not selected
    # by the action (in our problem)

    for e in ends:
        policy[e] = e

    for s in states:
        if not s in ends:
            ns = list(neighbors[s])
            best = V_curr[ns[0]]
            policy[s] = ns[0]

            # but we do have to loop over possible actions, which are one-to-one with s_prime
            # (argmax loop in wiki formula)
            for s_prime in ns:
                if V_curr[s_prime] > best:
                    best = V_curr[s_prime]
                    policy[s] = s_prime

    return policy

# TODO handle probability in way consistent w/ above
def update_values(values, states, policy, discount):
    # not a defaultdict anymore
    v_next = dict()

    for s in states:
        # TODO is this the right step 2?

        # only need policy and not neighbors, since the policy
        # produces a deterministic result (we don't need to sum
        # over all of the neighbors and weight be the probability
        # of our action taking us to that state)
        v_next[s] = V(s, policy, values, discount)

    return v_next

# the figure homework 7
states = {(0,0),(0,1),(0,2),(0,3),\
          (1,0),      (1,2),(1,3),\
          (2,0),(2,1),(2,2),(2,3)}

# default is zero, constructed this way
# WARNING: if you try to read any missing states,
# the dictionary may enter them mapped to the default value
# (so that it gets iterated over and 'in' checks pass)
# TODO are these initial values fine?
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

pi = initial_policy(neighbors, end_states)

'''
Implement a dynamic programming approach to solve this problem using policy it-
eration. Start with a random initial policy. Use a discount factor of γ = 0.99. For
each cell on the grid, display the value of the optimal value function V ∗ and show the
optimal policy π ∗ using arrows
'''

plot_states(states, values, start_state, end_states, pi)

policy_iterations = 1
value_iterations = 4

p = 0

# a set number of iterations, or until convergence of pi
while p < policy_iterations:
    # TODO not quite sure how to formulate this in terms of DP
    # so will first just try and implement equations in wiki on MDPs
    last_pi = pi
    pi = update_policy(values, states, neighbors, end_states)
    # check for policy convergence
    if pi == last_pi:
        print('Policy converged.')
        break

    v = 0

    last_values = None

    # either for a certain number of iterations or until convergence
    while v < value_iterations and not values_equal(last_values, values):
        # TODO no mutability issue right?
        last_values = values
        values = update_values(values, states, pi, discount)
        print('v:', v)
        print(values)
        if value_iterations < 5:
            plot_states(states, values, start_state, end_states, pi)
        v += 1


    print('p:', p)
    print(pi)
    print('')
    p += 1

plot_states(states, values, start_state, end_states, pi)
# TODO plot values too
