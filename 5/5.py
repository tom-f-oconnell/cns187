#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

sns.set()
# perhaps redundant
sns.set_style('dark')

should_plot = True
save_figs = False
verbose = False

figpath = './figures/'

plt.close('all')

def show_or_save(name):
    if save_figs:
        plt.savefig(figpath + name + '.eps', format='eps', dpi=1000)
        
    if should_plot:
        plt.show()
    else:
        plt.close()

def plot_classes(X, y, problem):
    plt.figure()

    if X.shape[0] == y.shape[0]:
        X_c0 = X[(y == 0).flatten(), :]
        X_c1 = X[(y == 1).flatten(), :]

        c0 = plt.scatter(X_c0[:,0], X_c0[:,1], c='r')
        c1 = plt.scatter(X_c1[:,0], X_c1[:,1],  c='b')
    else:
        X_c0 = X[:, (y == 0).flatten()]
        X_c1 = X[:, (y == 1).flatten()]

        c0 = plt.scatter(X_c0[0,:], X_c0[1,:], c='r')
        c1 = plt.scatter(X_c1[0,:], X_c1[1,:],  c='b')

    plt.title('Classes for dataset ' + problem)
    plt.legend([c0, c1], ['Class 0', 'Class 1'])

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.show()
    return xlim, ylim

def map_decisions(W, X, y, xlim, ylim, problem):

    # plot decision boundary
    samples = 200
    x = np.linspace(xlim[0], xlim[1], samples)
    y = np.linspace(ylim[0], ylim[1], samples)

    d = np.empty((samples, samples)) * np.nan
    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            d[i,j] = eval_net(W, np.array([[1], [xx], [yy]]), addones=False)

    #plt.figure()
    #plt.contour(x, y, d)

    #plt.matshow(d, cmap=plt.cm.viridis, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    plt.matshow(d, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])

    plt.title('Decision boundary for model in problem ' + problem)
    plt.colorbar()
    #plt.show()

    '''
    if X.shape[0] == y.shape[0]:
        X_c0 = X[(y == 0).flatten(), :]
        X_c1 = X[(y == 1).flatten(), :]

        c0 = plt.scatter(X_c0[:,0], X_c0[:,1], c='r')
        c1 = plt.scatter(X_c1[:,0], X_c1[:,1],  c='b')
    else:
        X_c0 = X[:, (y == 0).flatten()]
        X_c1 = X[:, (y == 1).flatten()]
    '''

    c0 = plt.scatter(X[0,(y==0).flatten()], X[1,(y==0).flatten()], c='r')
    c1 = plt.scatter(X[0,(y==1).flatten()], X[1,(y==1).flatten()],  c='b')

    plt.legend([c0, c1], ['Class 0', 'Class 1'])

    plt.show()

    return d

def logistic(W, X):
    return (1 / (1 + np.exp((-1) * np.dot(W, X.transpose())))).transpose()

def logistic(X):
    return (1 / (1 + np.exp((-1) * X.transpose()))).transpose()

def d_logistic(X):
    return logistic(X) * (1 - logistic(X))

def add_ones(X):
    '''
    Add the row of ones to the data matrix for the bias weights.
    '''

    ones = np.ones((1, X.shape[1]))
    return np.concatenate((ones, X), axis=0)

def eval_net(W, X, ret_intermediate=False, addones=True):
    '''
    Evaluate a matrix of input (example #, feature #) with weights matrix W defining
    a feedforward neural network.

    W must be indexed as (layer, unit # in input layer, unit # in current layer)
    '''

    # add the column of ones for the bias weights
    if addones:
        X = add_ones(X)

    # treat the data as the activations of a 0th layer of the network
    # allows consistent recursive definition
    g = X

    S = []
    G = []

    for l in range(0, len(W)):
        # calculate linear input to units in layer `l`
        S_l = np.dot(W[l].transpose(), g)
        
        if ret_intermediate:
            S.append(S_l)

        # pass the linearly weighted inputs through nonlinearity
        # which in this case is this logistic function
        g = logistic(S_l)

        if ret_intermediate:
            G.append(g)

    if ret_intermediate:
        # g (should be) = G[-1], but for convenience
        return (g, S, G)
    else:
        return g


def loss(y, y_est):
    '''
    Returns the squared error between provided estimates and true values.
    '''

    if y_est.shape == y.shape:
        return np.sum(np.square(y_est - y))
    elif y_est.shape == y.transpose().shape:
        return np.sum(np.square(y_est - y.transpose()))
    else:
        assert False
    
def backprop(W, X, y, y_est, S, G, rate):
    '''
    Uses the backpropagation algorithm to adjust the weights in place 
    by gradient descent.
    '''

    # dL/dw^l_{ij} = (dL/ds^l_j)(ds^l_j/dw^l_{ij})
    # ds^l_j/dw^l_{ij} = g^{l-1}_i, where g^l(x) = logistic((w^l)^T g^{l-1})
    # delta^l_j = (def) dL/ds^l_j
    
    # for the last layer; the base case
    # L in superscript is the last layer, otherwise L = Loss
    delta = 2 * (y_est - y.transpose()) * d_logistic(S[-1])

    # now recursively, for the other layers
    l = len(W) - 1
    while l > 0:
        # to make sure the dimensions don't change
        dims = [w.shape for w in W]

        # update the weights of the current layer
        # uses the same formula as the last layer does (could collapse)
        # what would happen if you had different learning rates for different layers?
        # work on this?
        W[l] = W[l] - (rate * np.dot(G[l-1], delta.transpose()))

        # the weights matrices should not have changed dimensions
        for w, d in zip(W, dims):
            try:
                assert w.shape == d
            except AssertionError:
                print(str(w.shape) + ' differed from previous ' + str(d))
                assert False

        # actually propagate back one layer
        # i.e. use the delta from the previous layer
        # to calculate the delta for the current layer
        sum_term = np.dot(W[l], delta)

        # delta for layer l-1
        delta = d_logistic(S[l-1]) * sum_term

        # now move one layer closer to the input
        l = l - 1
    
    return None

def train_net(X, y, dims, iterations):
    '''
    Train a neural network with number of units in all hidden layers
    determined by the corresponding element in dims.

    Do not include the dims of the input or output layers in dims.

    Returns the weights for each layer (apart from the first = data).
    '''
    '''
    # only for current version of cross-entropy cost function
    y = y.astype(dtype=np.int16)
    y[y == 0] = -1
    '''

    print('Training network...')
    # add the column of ones for the bias weights
    X = add_ones(X)

    # initial parameter guesses (stored as a list of matrices)
    # free parameters are the bias and input component weights

    # len(dims) + 1 because we need a transform from the last hidden
    # to the output, which can not have a variable number of units
    # since it must be the dimensions of the output

    W = []
    # TODO will need to re-encode multiclass output in one-hot format or this wont work
    dims = [X.shape[0]] + dims + [y.shape[1]]

    # initialize all weights to random values
    for l in range(0, len(dims) - 1):
        W_l = np.random.rand(dims[l], dims[l+1])
        W.append(W_l)

    # learning rate
    # could put on a schedule
    # 0.01 is too high. oscillations.
    rate = 0.001

    last_loss = np.nan
    loss_t = np.zeros(iterations)
    t = 0
        
    # fit the parameters by repeating forward -> backprop (gradient descent)
    while t < iterations:

        # forward prop for our estimate
        y_est, S, G = eval_net(W, X, ret_intermediate=True, addones=False)
        L = loss(y, y_est)

        loss_t[t] = L

        # weights adjusted in place
        backprop(W, X, y, y_est, S, G, rate)

        t = t + 1

        if t % (iterations / 100) == 0:
            print('t=' + str(t))
            print('loss=' + str(L))
            print('accuracy=' + str(accuracy(y_est, y)))

    y_est, S, G = eval_net(W, X, ret_intermediate=True, addones=False)
    return W, loss_t, y_est

def accuracy(y_est, y):
    # abstracts away some of the sign convention stuff

    if np.any(y == -1):
        return np.sum(np.sign(y_est) == y) / np.size(y)
    else:
        if y_est.shape == y.shape:
            return np.sum(np.round(y_est) == y) / np.size(y)
        elif y.transpose().shape == y_est.shape:
            return np.sum(np.round(y_est) == y.transpose()) / np.size(y)
        else:
            assert False

'''
Problem 1
'''

Data = scipy.io.loadmat('dataset.mat')

# want dimensions to go as (# features, # examples)
Xtr = Data['Xtr']
ytr = Data['ytr']
Xts = Data['Xts']
yts = Data['yts']

"""
print('1.1')

# doesnt count input (data) or output layer
# because those are determined by dimensions of X and y
dims = [5, 5]
W, loss_t, y_eest = train_net(Xtr, ytr, dims, 50000)

plt.plot(loss_t)
plt.title('Loss over time for 1.1')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

y_est = eval_net(W, Xtr)

# TODO remove
# assert np.allclose(y_eest, y_est, rtol=0.01, atol=0.02)

y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

d = map_decisions(W, Xtr, ytr, xlim, ylim, '1.1')

print('1.2')
plt.figure()

dims = [2]
W, loss_t, y_eest = train_net(Xtr, ytr, dims, 50000)

plt.plot(loss_t)
plt.title('Loss over time for 1.2')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

d = map_decisions(W, Xtr, ytr, xlim, ylim, '1.2')
"""
# scatter plot to compare all decision heatmaps against
xlim, ylim = plot_classes(Xtr, ytr, '1')

print('1.4')
plt.figure()

dims = [4, 3]
# TODO training for 10^6 iterations eventually leads to oscillations 
# around 32-33 loss. why?
# in loss or actually just in accuracy? should cost func explicitly model the latter?
W, loss_t, y_eest = train_net(Xtr, ytr, dims, 100000)

print(Xtr.shape)

plt.plot(loss_t)
plt.title('Loss over time for 1.4')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

print(len(W))
print(Xtr.shape)

# TODO why are these values different from values returned from training?
y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

# TODO is testing always better than train??? it was this time... by 2.4% points as well
print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

d = map_decisions(W, Xtr, ytr, xlim, ylim, '1.4')

"""
Problem 2
"""
'''
Train = scipy.io.loadmat('mnist_train.mat')
Test = scipy.io.loadmat('mnist_test.mat')

# want dimensions to go as (# features, # examples)
Xtr = Train['data']
ytr = Train['labels']
Xts = Test['data']
yts = Test['labels']
'''
