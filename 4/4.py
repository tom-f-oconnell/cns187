#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

""" Problem 2 """

or_x = np.array([1, 1, 0, 0])
or_y = np.array([1, 0, 1, 0])
or_output = np.array([1, 1, 1, 0])

# 2.1
# TODO change colormap or line thickness (0 is hardly visible)
plt.scatter(or_x, or_y, c=or_output)
# TODO make bold
plt.xlabel('x1 truth value')
plt.ylabel('x2 truth value')
plt.title('OR function of x1 and x2')
# TODO legend

# 2.2
# (of course this is not the only correct decision boundary to 
#  separate output of the OR function)
plt.plot([-0.2, 1], [1, -0.2], 'r-')
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
#plt.show()

# 2.3
# added dimension is for the bias term
# both the weights and the biases will be initialized to zeros
# could also use (small) random numbers
weights = np.zeros(3)

training_set = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
# stack a column vector of ones of the left
# for the bias term
X = np.concatenate((np.ones((training_set.shape[0], 1)), training_set), axis=1)

print('Training set:')
print(training_set)
print('')

print('X:')
print(X)
print('')

def nonlinearity(x):
    if x > 0:
        return 1
    else:
        return 0

# TODO how to make a table in latex?
print('x1, x2, y, w1, w2, b')
print('--, --, -,  0,  0, 0')
updated_weights = True
i = 0
while updated_weights and i < X.shape[0]:
    # calculate the output with current weights
    y_est_i = nonlinearity(np.dot(weights, X[i,:-1]))
    # update the weights
    weights = weights + (X[i,3] - y_est_i) * X[i, 0:3]

    # skipping the first element of the training set
    # because that is the bias
    print(int(X[i,1]), ' ', int(X[i,2]), ' ', int(X[i,3]), ' ', \
            weights[1], weights[2], weights[0])

    i = i + 1


""" Problem 3 """

