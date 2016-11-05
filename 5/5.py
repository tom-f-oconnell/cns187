#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

sns.set()

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

def map_decisions(W, X, y):
    fig = plt.figure()

    if X.shape[0] == y.shape[0]:
        X_c0 = X[(y == 0).flatten(), :]
        X_c1 = X[(y == 1).flatten(), :]

        plt.scatter(X_c0[:,0], X_c0[:,1], c='r')
        plt.scatter(X_c1[:,0], X_c1[:,1],  c='b')
    else:
        X_c0 = X[:, (y == 0).flatten()]
        X_c1 = X[:, (y == 1).flatten()]

        plt.scatter(X_c0[0,:], X_c0[1,:], c='r')
        plt.scatter(X_c1[0,:], X_c1[1,:],  c='b')

    # plot decision boundary
    ax = plt.gca()
    prev_x = ax.get_xlim()
    prev_y = ax.get_ylim()

    samples = 200
    x = np.linspace(prev_x[0], prev_x[1], samples)
    y = np.linspace(prev_y[0], prev_y[1], samples)
    '''
    xx, yy = np.meshgrid(x, y)

    #d = np.round(1 / (1 + np.exp(W[0,0] + W[0,1]*xx + W[0,2]*yy + \
    #        W[0,3]*xx*yy + W[0,4]*(xx**2) + W[0,5]*(yy**2))))
    print([xx,yy])
    print(xx.shape)
    print(yy.shape)

    xy = np.concatenate((xx,yy), axis=0)
    '''

    d = np.zeros((samples, samples))
    for xx in x:
        for yy in y:
            # TODO dimension
            d[xx,yy] = eval_net(W, np.array([1, xx, yy]), addones=False)

    plt.contour(x, y, d, 1, colors='g')
    plt.show()

    return fig, xx, yy

def logistic(W, X):
    # TODO why the final transpose? fix?
    # TODO should hopefully have to change dot product
    return (1 / (1 + np.exp((-1) * np.dot(W, X.transpose())))).transpose()

def logistic(X):
    # TODO if we change above, change this
    return (1 / (1 + np.exp((-1) * X.transpose()))).transpose()

# TODO check
def d_logistic(X):
    return logistic(X) * (1 - logistic(X))

def add_ones(X):
    # TODO change to make math line up in eval_net if needed
    '''
    Add the row of ones to the data matrix for the bias weights.
    '''

    print(X.shape)
    ones = np.ones((1, X.shape[1]))
    print(ones.shape)
    #worked with transposed X
    #return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return np.concatenate((ones, X), axis=0)

def eval_net(W, X, ret_intermediate=False, addones=True):
    # TODO check all indexing
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
    #print('backprop')

    # dL/dw^l_{ij} = (dL/ds^l_j)(ds^l_j/dw^l_{ij})
    # ds^l_j/dw^l_{ij} = g^{l-1}_i, where g^l(x) = logistic((w^l)^T g^{l-1})
    # delta^l_j = (def) dL/ds^l_j
    
    # TODO correct subscripts?
    # for the last layer; the base case
    # L in superscript is the last layer, otherwise L = Loss
    '''
    # L = (g^L_i - y_i)^2 => dL/dg^l_j = 2*(g^L_i - y_i)
    # delta^L_i = dL/ds^L_i
    # TODO make sure there is an S for the last layer
    # TODO abs?
    # TODO TODO TODO this has got to be wrong
    #delta = np.sum(2 * (y_est - y.transpose()) * d_logistic(S[-1]))
    '''
    delta = 2 * (y_est - y.transpose()) * d_logistic(S[-1])

    '''
    print(np.sum(np.abs(y_est - y.transpose())))
    print(y_est.shape)
    print(y.shape)
    print(d_logistic(S[-1]).shape)
    #print("first delta " + str(delta.shape))

    # update the weights for the last layer
    # ds^l_j/dw^l_{ij} = g^L_j (and in general?)
    print(W[-1].shape)
    print((rate * delta * G[-1]).shape)
    print(G[-1].shape)
    print(delta.shape)
    print(np.dot(delta, G[-1]).shape)
    W[-1] = W[-1] - rate * np.dot(delta, G[-1])
    print(W[-1].shape)
    '''

    # now recursively, for the other layers
    l = len(W) - 1
    # TODO >= 0 or > 0? should i be doing the weight update above?
    while l > 0:
        #print('LAYER=' + str(l))

        # to make sure the dimensions don't change
        dims = [w.shape for w in W]

        # update the weights of the current layer
        # uses the same formula as the last layer does (could collapse)
        # what would happen if you had different learning rates for different layers?
        # work on this?
        #W[l-1] = W[l-1] - rate * delta * G[l-1]
        '''
        print("delta " + str(delta.shape))
        print(G[l].shape)
        print("G[l-1] " + str(G[l-1].shape))
        print((delta * G[l-1]).shape)
        print((rate * np.sum(delta * G[l-1], axis=1)).shape)
        print(W[l].shape)

        print(W[l].size)
        print((rate * np.sum(delta * G[l-1], axis=1)).size)
        '''
        #W[l] = W[l] - (rate * np.sum(delta * G[l-1], axis=1)).reshape(W[l].shape)
        # TODO check size
        # the delta is from l-1
        # TODO how to make G be from one layer less than l?
        #W[l] = W[l] - (rate * np.dot(delta, G[l-1]))
        W[l] = W[l] - (rate * np.dot(G[l-1], delta.transpose()))

        # the weights matrices should not have changed dimensions
        for w, d in zip(W, dims):
            try:
                assert w.shape == d
            except AssertionError:
                print(str(w.shape) + ' differed from previous ' + str(d))
                assert False

        #print('just updated weights')

        # actually propagate back one layer
        # i.e. use the delta from the previous layer
        # to calculate the delta for the current layer
        sum_term = np.zeros((W[l].shape[0], y.shape[0]))
        #print(sum_term.shape)

        # TODO vectorize
        # shape[1] should be the number of units 'in' layer l (the output of)
        # TODO check
        #print("W[l] shape")
        #print(W[l].shape)
        for j in range(0, W[l].shape[1]):
            '''
            print(j)
#            try:
            print(delta[j].shape)
            print(delta.shape)
            '''
            #sum_term = sum_term + delta[j] * W[l][:,j]
            sum_term = sum_term + np.outer(delta[j], W[l][:,j]).transpose()
        '''
                    except IndexError:
                        #sum_term = sum_term + delta * W[l][:,j]
                        print(delta.shape)
                        print(W[l].shape)
                        print(W[l][:,j].shape)
                        print(np.outer(delta, W[l][:,j]).transpose().shape)

                        sum_term = sum_term + np.outer(delta, W[l][:,j]).transpose()
                # TODO again. must sum be wrong? maybe do iteratively instead? SGD?
                #delta = np.sum(d_logistic(S[l-1]), axis=1) * sum_term

        print('')

        print('sumterm shape ' + str(sum_term.shape))
        '''
        # delta for layer l-1
        delta = d_logistic(S[l-1]) * sum_term

        # now move one layer closer to the input
        l = l - 1
    
    #print('leaving backprop')
    return None

def train_net(X, y, dims):
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
    # dims.append(y.shape[1])
    dims = [X.shape[0]] + dims + [y.shape[1]]
    #print(dims)

    # initialize all weights to random values
    for l in range(0, len(dims) - 1):

        # TODO random enough? should they all be positive (subtract 0.5?)?
        W_l = np.random.rand(dims[l], dims[l+1])
        #print(W_l.shape)

        W.append(W_l)

    # learning rate
    # could put on a schedule
    # 0.01 is too high. oscillations.
    rate = 0.001

    last_loss = np.nan

    # TODO remove
    N = X.shape[0]
    print('N=' + str(N))

    iterations = 50000

    loss_t = np.zeros(iterations)

    # fit the parameters by repeating forward -> backprop (gradient descent)
    # TODO plot loss over time
    #while True:
    
    t = 0
        
    while t < iterations:

        # forward prop for our estimate
        y_est, S, G = eval_net(W, X, ret_intermediate=True, addones=False)
        L = loss(y, y_est)

        loss_t[t] = L

        # weights adjusted in place
        # TODO verify that this is the case
        backprop(W, X, y, y_est, S, G, rate)

        #if verbose:
        print('t=' + str(t))
        print('loss=' + str(L))
        print('accuracy=' + str(accuracy(y_est, y)))

        '''
        if np.isnan(L) or np.isclose(last_loss, L):
            break
        last_loss = L
        '''
        
        t = t + 1

    return W, loss_t

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
print('1.1')

Data = scipy.io.loadmat('dataset.mat')

# want dimensions to go as (# features, # examples)
Xtr = Data['Xtr']
ytr = Data['ytr']
Xts = Data['Xts']
yts = Data['yts']

# doesnt count input (data) or output layer
# because those are determined by dimensions of X and y
dims = [5, 5]
W, loss_t = train_net(Xtr, ytr, dims)

plt.plot(loss_t)
plt.title('Loss over time for 1.1')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

mf, xx, yy = map_decisions(W, Xtr, ytr)

print('1.2')
plt.figure()

dims = [2]
W, loss_t = train_net(Xtr, ytr, dims)

plt.plot(loss_t)
plt.title('Loss over time for 1.1')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

mf, xx, yy = map_decisions(W, Xtr, ytr)

print('1.4')
plt.figure()

dims = [5, 2]
W, loss_t = train_net(Xtr, ytr, dims)

plt.plot(loss_t)
plt.title('Loss over time for 1.1')
plt.xlabel('Iteration number')
plt.ylabel('Squared error loss function')
plt.show()

y_est = eval_net(W, Xtr)
y_est_ts = eval_net(W, Xts)

print('Training set accuracy=' + str(accuracy(y_est, ytr)))
print('Testing set accuracy=' + str(accuracy(y_est_ts, yts)))

f, xx, yy = map_decisions(W, Xtr, ytr)

