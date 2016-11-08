#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import time

import pickle

sns.set()
# perhaps redundant
sns.set_style('dark')

# these two are not used right now
should_plot = True
save_figs = False

verbose = True

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

    c0 = plt.scatter(X[0,(y==0).flatten()], X[1,(y==0).flatten()], c='r')
    c1 = plt.scatter(X[0,(y==1).flatten()], X[1,(y==1).flatten()],  c='b')

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

    # not working here either. sigh.
    #c0 = plt.scatter(X[0,np.where(y==0)], X[1,np.where(y==0)], c='r')
    #c1 = plt.scatter(X[0,np.where(y==1)], X[1,np.where(y==1)],  c='b')

    #plt.matshow(d, cmap=plt.cm.viridis, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')
    plt.matshow(d, alpha=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')
    plt.colorbar()

    plt.title('Decision boundary for model in problem ' + problem)

    '''
    c0 = plt.scatter(X[0,(y==0).flatten()], X[1,(y==0).flatten()], c='r')
    c1 = plt.scatter(X[0,(y==1).flatten()], X[1,(y==1).flatten()],  c='b')
    '''

    #plt.legend([c0, c1], ['Class 0', 'Class 1'])

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
        print(y.shape)
        print(y_est.shape)
        # TODO raise appropriate exception instead
        assert False

def expand_images(X):
    """
    Convert 28 x 28 MNIST images into vectors
    """

    X_ex = np.empty((X.shape[0] * X.shape[1], X.shape[2])) * np.nan

    for n in range(0, X.shape[2]):
        X_ex[:,n] = X[:,:,n].flatten()

    return X_ex

def shuffle_data(X):
    """
    Randomly re-order datapoints.
    """
    # MNIST data might have already been shuffled though

    X_rand = X[:, np.random.randint(0,X.shape[1])]
    return X_rand


def one_hot_encoding(y):
    """
    Change y to a {0/1}^d representation as opposed to integers on [0,d]
    """

    y_oh = np.zeros((y.shape[0], y.max() - y.min() + 1))

    # currently only works in min is actually 0
    for j in range(0, y_oh.shape[1]):
        y_oh[np.where(y == j), j] = 1

    return y_oh

    
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

    '''
    print('delta before loop')
    tmp = delta
    print(tmp.mean())
    print(tmp.var())
    print(np.sum(np.square(tmp))) 
    print('')

    # these were the (~0) terms
    print(S[-1].mean())
    print(S[-1].var())
    print(d_logistic(S[-1]).mean())
    print(d_logistic(S[-1]).var())
    print((y_est - y.transpose()).mean())
    print((y_est - y.transpose()).var())

    print('')
    '''

    # now recursively, for the other layers
    l = len(W) - 1
    while l > 0:
        # to make sure the dimensions don't change
        # dims = [w.shape for w in W]

        # update the weights of the current layer
        # uses the same formula as the last layer does (could collapse)
        # what would happen if you had different learning rates for different layers?
        # work on this?
        
        # this actually does fail
        #assert np.allclose(rate * np.dot(G[l-1], delta.transpose()), np.zeros(W[l].shape))

        # TODO these quickly all go to zero (after 1 or 2 iteration)
        # (not with learning rate ~0.001 though)
        '''
        print('weight update')
        tmp = rate * np.dot(G[l-1], delta.transpose())
        print(tmp.mean())
        print(tmp.var())
        print(np.sum(np.square(tmp))) 
        print('')
        '''

        W[l] = W[l] - (rate * np.dot(G[l-1], delta.transpose()))

        ''' they haven't been ''
        # the weights matrices should not have changed dimensions
        for w, d in zip(W, dims):
            try:
                assert w.shape == d
            except AssertionError:
                print(str(w.shape) + ' differed from previous ' + str(d))
                assert False
        '''

        # actually propagate back one layer
        # i.e. use the delta from the previous layer
        # to calculate the delta for the current layer
        sum_term = np.dot(W[l], delta)

        '''
        print('sum_term')
        tmp = sum_term
        print(tmp.mean())
        print(tmp.var())
        print(np.sum(np.square(tmp))) 
        print('')
        '''

        # delta for layer l-1
        delta = d_logistic(S[l-1]) * sum_term
        
        '''
        print('delta')
        tmp = delta
        print(tmp.mean())
        print(tmp.var())
        print(np.sum(np.square(tmp))) 
        print('')
        '''

        # now move one layer closer to the input
        l = l - 1


def train_net(X, y, dims, iterations, rate=0.01, batch=None, decay=None):
    '''
    Train a neural network with number of units in all hidden layers
    determined by the corresponding element in dims.

    Do not include the dims of the input or output layers in dims.

    Returns the weights for each layer (apart from the first = data).
    '''

    # add the column of ones for the bias weights
    X = add_ones(X)

    # initial parameter guesses (stored as a list of matrices)
    # free parameters are the bias and input component weights

    # len(dims) + 1 because we need a transform from the last hidden
    # to the output, which can not have a variable number of units
    # since it must be the dimensions of the output

    W = []
    dims = [X.shape[0]] + dims + [y.shape[1]]
    print('Training network with dims ' + str(dims) + '...')
    print(str(iterations) + ' iterations at ' + str(rate) + ' learning rate.')
    if not batch is None:
        print('Using batch size of ' + str(batch))
    if not decay is None:
        print('Decaying learning rate every ' + str(iterations / 1000) + \
                ' iterations by factor of ' + str(decay)) 

    # initialize all weights to random values
    for l in range(0, len(dims) - 1):
        W_l = np.random.rand(dims[l], dims[l+1]) - 0.5
        W.append(W_l)

    # learning rate
    # could put on a schedule
    # 0.01 is too high. oscillations.
    # (a kwarg now)

    last_loss = np.nan
    loss_t = np.zeros(iterations)
    t = 0
        
    # fit the parameters by repeating forward -> backprop (gradient descent)
    while t < iterations:

        if batch is None:
            # forward prop for our estimate
            y_est, S, G = eval_net(W, X, ret_intermediate=True, addones=False)

            # weights adjusted in place
            backprop(W, X, y, y_est, S, G, rate)

            L = loss(y, y_est)
            loss_t[t] = L
        else:
            
            # TODO fix boundary problems where less samples are taken at end
            start = ((t * batch) % X.shape[1])
            last = start + batch
            y_est, S, G = eval_net(W, X[:,start:last].reshape((X.shape[0],batch)) \
                    , ret_intermediate=True, addones=False)

            backprop(W, X[:,start:last].reshape((X.shape[0],batch)), \
                    y[start:last,:].reshape((batch,y.shape[1])), y_est, S, G, rate)

        if verbose and t % (iterations / 100) == 0:

            print('t=' + str(t))

            y_est = eval_net(W, X, ret_intermediate=False, addones=False)
            L = loss(y, y_est)
            loss_t[t] = L
            
            print('rate=' + str(rate))
            print('loss=' + str(L))
            print('accuracy=' + str(accuracy(y_est, y)))
            print('')

            if not decay is None:
                # TODO better schedule?
                rate = decay * rate

        t = t + 1

    y_est, S, G = eval_net(W, X, ret_intermediate=True, addones=False)
    return W, loss_t, y_est

def accuracy(y_est, y):
    # abstracts away some of the sign convention stuff

    # check for one hot encoding
    if y_est.shape[0] > 1:
        if y_est.shape == y.shape:
            return np.sum(np.argmax(y_est, axis=1) == np.argmax(y, axis=1)) / y_est.shape[0]
        elif y.transpose().shape == y_est.shape:
            return np.sum(np.argmax(y_est, axis=0) == np.argmax(y.transpose(), axis=0)) \
                    / y_est.shape[1]
        else:
            assert False
    else:
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
# TODO train might actually be worse than test sometimes...
# why? not optimizing correctly? a function of the way the two were created?

Data = scipy.io.loadmat('dataset.mat')

# want dimensions to go as (# features, # examples)
Xtr = Data['Xtr']
ytr = Data['ytr']
Xts = Data['Xts']
yts = Data['yts']
'''
print('1.1')

# scatter plot to compare all decision heatmaps against
xlim, ylim = plot_classes(Xtr, ytr, '1')

# doesnt count input (data) or output layer
# because those are determined by dimensions of X and y
dims = [5, 5]
W, loss_t, y_eest = train_net(Xtr, ytr, dims, 20000)

plt.figure()
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

d1 = map_decisions(W, Xtr, ytr, xlim, ylim, '1.1')

print('1.2')
plt.figure()

dims = [2]
W, loss_t, _ = train_net(Xtr, ytr, dims, 50000)

plt.plot(loss_t)
plt.title('Loss over time for 1.2')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

d2 = map_decisions(W, Xtr, ytr, xlim, ylim, '1.2')

print('1.4')

dims = [50]
# TODO training for 10^6 iterations eventually leads to oscillations 
# around 32-33 loss. why?
# in loss or actually just in accuracy? should cost func explicitly model the latter?
W, loss_t, _ = train_net(Xtr, ytr, dims, 80000)

plt.figure()
plt.plot(loss_t)
plt.title('Loss over time for 1.4')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

# TODO why are these values different from values returned from training?
# check, but they should be the same now
y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

d4 = map_decisions(W, Xtr, ytr, xlim, ylim, '1.4')

'''
"""
Problem 2
"""

Train = scipy.io.loadmat('mnist_train.mat')
Test = scipy.io.loadmat('mnist_test.mat')

# want dimensions to go as (# features, # examples)
Xtr = Train['data']
ytr = Train['labels']
Xts = Test['data']
yts = Test['labels']

# expand the images into vectors of features
Xtr_ex = expand_images(Xtr)
Xts_ex = expand_images(Xts)

# recode the output as vectors of 0 and 1
ytr_oh = one_hot_encoding(ytr)
yts_oh = one_hot_encoding(yts)

use_saved = True

if not use_saved:
    dims = [784, 11]
    W, loss_t, _ = train_net(Xtr_ex, ytr_oh, dims, 40000, rate=0.01, batch=50, decay=0.99)

    print('Saving weights and loss during training to two separate .npy files.')

    np.save('W.npy', W)
    np.save('loss_t.npy', loss_t)
else:
    print('Loading saved weights and loss during training.')

    with open('W.npy', 'rb') as f:
        W = np.load(f)
    with open('loss_t.npy', 'rb') as f:
        loss_t = np.load(f)

y_est = eval_net(W, Xtr_ex)
y_est_ts = eval_net(W, Xts_ex)

print('Training set accuracy=' + str(accuracy(y_est, ytr_oh)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts_oh)))

classes = ytr_oh.shape[1]

# frequency over true class by predicted class dimensions
Confusion = np.empty((classes,classes)) * np.nan

for c in range(0,classes):
    y = eval_net(W, Xtr_ex[:, (ytr == c).flatten()])
    # first index is true class
    # unnormalized
    Confusion[c,:] = np.sum(y, axis=1)

# log scale maybe?
plt.matshow(Confusion, cmap=plt.cm.viridis)
plt.title('Confusion matrix for MNIST classifier')
# TODO check these are right
plt.xlabel('True class')
plt.ylabel('Predicted class')
# TODO normalize?
plt.colorbar()
plt.show()
