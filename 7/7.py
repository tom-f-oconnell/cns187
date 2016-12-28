#!/usr/bin/env python3

import numpy as np

"""
Policy iteration and Q-learning practice
"""

def neighbors(s, states):
    """ Returns a set of states s' that are neighbors of s in states """

    i, j = s
    ns = set()

    for s_prime in states:
        i_prime, j_prime = s_prime

        # we don't want it to be the same square
        # or anything further than one taxicab distance away
        if abs(i - i_prime) + abs(j - j_prime) == 1:
            ns.add(s_prime)
        
    return ns

def initial_policy(states):
    """ Generate a random initial policy uniformly """

    policy = dict()
    for s in states:
        # pick a random next state
        # so our initial policy is defined everywhere
        # needs to be a neighbor of s
        policy[s] = np.random.choice(neighbors(s, states))

    return policy

# the figure homework 7
states = {(0,0),(0,1),(0,2),(0,3),\
          (1,0),      (1,2),(1,3),\
          (2,0),(2,1),(2,2),(2,3)}

start_state = (0,0)
# TODO values in this, or more generally somewhere?
end_states = {(1,2),(2,3)}

pi = initial_policy(states)

discount = 0.99

'''
Implement a dynamic programming approach to solve this problem using policy it-
eration. Start with a random initial policy. Use a discount factor of γ = 0.99. For
each cell on the grid, display the value of the optimal value function V ∗ and show the
optimal policy π ∗ using arrows
'''

# TODO not quite sure how to formulate this in terms of DP
# so will first just try and implement equations in wiki on MDPs
pi = update_policy()
V = update_values()
