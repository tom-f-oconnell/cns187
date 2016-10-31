#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

sns.set()

should_plot = True
verbose = False

def plot_by_class(X, y):

    plt.figure()

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

    plt.scatter(X_c0[0,:], X_c0[1,:], c='r')
    plt.scatter(X_c1[0,:], X_c1[1,:],  c='b')

""" Problem 1.2 """

or_x = np.array([1, 1, 0, 0])
or_y = np.array([1, 0, 1, 0])
or_output = np.array([1, 1, 1, 0])

# 1.2.1
print('Problem 1.2.1')

if should_plot:
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

# 1.2.3
# added dimension is for the bias term
# both the weights and the biases will be initialized to zeros
# could also use (small) random numbers
weights = np.zeros(3)

features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
labels = np.array([[0],[1],[1],[1]])
training_set = np.concatenate((features, labels), axis=1)
#training_set = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

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
    ''' Receives w*x + b as input '''

    if x > 0:
        return 1
    else:
        return 0

updated_weights = True
curr = 0
print('x1  x2  y   w1   w2  w0')
while curr < 10:
    i = curr % X.shape[0]

    if verbose:
        print('')
        print('ITERATION ' + str(curr))

    # calculate the output with current weights
    y_est_i = nonlinearity(np.dot(weights, X[i,:-1]))

    if verbose:
        print('Weights before: ' + str(weights))
        print('Truth : ' + str(X[i,3]))
        print('Estimate : ' + str(y_est_i))
        print('Activation before :')
        print(np.dot(weights, X[i, 0:3]))
        print(nonlinearity(np.dot(weights, X[i, 0:3])))

    # update the weights
    # if the estimate is correct, this won't change the weights
    weights = weights + (X[i,3] - y_est_i) * X[i, 0:3]

    if verbose:
        print('Weights after: ' + str(weights))
        print('Activation after :')
        print(np.dot(weights, X[i, 0:3]))
        print(nonlinearity(np.dot(weights, X[i, 0:3])))

        if X[i,3] == y_est_i:
            print('CORRECT')

    if curr < 4:
        # skipping the first element of the training set
        # because that is the bias
        print(int(X[i,1]), ' ', int(X[i,2]), ' ', int(X[i,3]), ' ', \
                weights[1], weights[2], weights[0])

    if verbose:
        print('')

    curr = curr + 1

# check the output
if verbose:
    print(np.dot(weights, X[:,:-1].transpose()))
    print(np.sign(np.dot(weights, X[:,:-1].transpose())))

# evaluates the decision boundary at x2 to find the corresponding x1
bound_x2y = lambda x2: (-weights[0] - weights[2]*x2) / weights[1]
bound_y2x = lambda x1: (-weights[0] - weights[1]*x1) / weights[2]

# now plot the learned decision boundary
# decision boundary: x1 = (-w0 - w2*x2) / w1
x = -0.2
p_0 = (x, bound_x2y(x))
y = -0.2
p_1 = (bound_y2x(y), y)

if should_plot:
    plt.plot(p_0, p_1, 'g-')
    plt.show()

""" Problem 1.3 """

print('')
print('Problem 1.3')

'''
Loads variables Xtr, ytr, Xts, yts
which are the training and testing sets, respectively.
'''
A = scipy.io.loadmat('datasetA.mat')
Xtr = A['Xtr']
ytr = A['ytr']
Xts = A['Xts']
yts = A['yts']

if should_plot:
    plt.figure()

# 1.3.a: plot the data and draw linear boundaries that could collectively
# be used to classify the data (through ANDS and ORs)

Xtr_c0 = Xtr[:, (ytr == 0).flatten()]
Xtr_c1 = Xtr[:, (ytr == 1).flatten()]
plt.scatter(Xtr_c0[0,:], Xtr_c0[1,:], c='r')
plt.scatter(Xtr_c1[0,:], Xtr_c1[1,:],  c='b')

# I manually picked a series of points around the perimeter of each 
# of the regions that the class 1 points appear to lie in
ps_a = np.array([[-1.51,0.98], [-1.46,-0.86], [-0.77,-1.26], \
       [-0.35,-0.02], [-0.81,1.07]])

# just need to adjust these until test points lead to positives
# in the first five (length of this list) linear threshold units
flip_sign_a = [True, True, True, False, False]

ps_b = np.array([[0.39,0.75], [0.45,-1.13], [1.00, -1.17], \
       [1.28,-0.84], [1.35,0.16], [1.23,0.84]])

# likewise, adjust these until test points in right region
# are position for each of the last six linear threshold units
flip_sign_b = [True, True, True, True, False, False]

W_0 = np.empty((3, len(ps_a) + len(ps_b))) * np.nan
curr = 0

for i in range(0, len(ps_a)):
    # get pair of points with neighboring index (modulo length of list)
    if i < len(ps_a) - 1:
        pair = ps_a[i:i+2, :]
    else:
        pair = np.array([ps_a[i, :], ps_a[0, :]])

    if should_plot:
        plt.plot(pair[:,0], pair[:,1], 'g-')
        offset = 0.1
        plt.text(np.mean(pair[:,0]) - offset, np.mean(pair[:,1]), str(curr), size=15)

    # calculate linear threshold unit params for line along pair of points
    w1 = (pair[1,1] - pair[0,1]) / (pair[1,0] - pair[0,0])
    w0 = pair[1,1] - w1 * pair[1,0]
    w2 = 1

    # TODO check some test points still fall on the lines
    # check correct classification
    
    if flip_sign_a[i]:
        W_0[:, curr] = [-w0, -w1, -w2]
    else:
        W_0[:, curr] = [w0, w1, w2]

    curr = curr + 1

for i in range(0, len(ps_b)):
    if i < len(ps_b) - 1:
        pair = ps_b[i:i+2, :]
    else:
        pair = np.array([ps_b[i, :], ps_b[0, :]])

    if should_plot:
        plt.plot(pair[:,0], pair[:,1], 'g-')
        offset = 0.1
        plt.text(np.mean(pair[:,0]) - offset, np.mean(pair[:,1]), str(curr), size=15)

    # calculate linear threshold unit params for line along pair of points
    w1 = (pair[1,1] - pair[0,1]) / (pair[1,0] - pair[0,0])
    w0 = pair[1,1] - w1 * pair[1,0]
    w2 = 1

    if flip_sign_b[i]:
        W_0[:, curr] = [-w0, -w1, -w2]
    else:
        W_0[:, curr] = [w0, w1, w2]

    curr = curr + 1

if should_plot:
    plt.title('Linear decision boundaries circling class 1')
    plt.show()

# 1.3.b: drawn and included in pdf

# 1.3.c: manually implement the LTU -> AND -> OR classifier for this data

def manual_AND_OR_est(X, nonlin=np.sign):
    # add the column of ones for the bias weights
    X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    ltu = nonlin(np.dot(W_0.transpose(), X))
    if verbose:
        print(ltu)

    g_left = np.concatenate((np.ones((1, X.shape[1])), ltu[:len(ps_a),:]), axis=0)
    g_right = np.concatenate((np.ones((1, X.shape[1])), ltu[len(ps_a):,:]), axis=0)

    Left_AND = np.zeros(len(ps_a) + 1)
    Right_AND = np.zeros(len(ps_b) + 1)
    epsilon = 1e-8

    # weights to AND all of the LTUs that circle the left region
    # will take all inputs being 1 to let the sum of this transformation equal or exceed 1
    Left_AND[0] = -1
    Left_AND[1:] = 1 / len(ps_a) + epsilon

    # weights to AND all of the LTUs that circle the right region
    Right_AND[0] = -1
    Right_AND[1:] = 1 / len(ps_b) + epsilon

    if verbose:
        print(np.dot(Left_AND, g_left))
        print(np.dot(Right_AND, g_right))

    Left_out = nonlin(np.dot(Left_AND, g_left))
    Right_out = nonlin(np.dot(Right_AND, g_right))

    if verbose:
        print(Left_out)
        print(Right_out)

    g_and = np.concatenate((np.ones((1,X.shape[1])), np.array([Left_out, Right_out])), axis=0)

    if verbose:
        print(g_and)

    # one negative label (-1) should not be able to pull
    # unit below threshold (hence bias weight > input unit weight)
    OR = np.array([1.5, 1, 1])

    if verbose:
        print(np.dot(OR, g_and))

    # switch back to the sign convention of the loaded dataset
    tmp = nonlin(np.dot(OR, g_and))
    tmp[tmp == -1] = 0
    return tmp

# both bad: 0, 0
# left good: -1, 0
# right good: 1, 0
#test_est = manual_AND_OR_est(np.array([[0, 0],[-1, 0],[1, 0]]).transpose())

ytr_est = manual_AND_OR_est(Xtr)
yts_est = manual_AND_OR_est(Xts)

Xtr_c0_est = Xtr[:, (ytr_est == 0).flatten()]
Xtr_c1_est = Xtr[:, (ytr_est == 1).flatten()]
# TODO i'm confused, because this has points, but the above seems to
# show them as correctly classified
Xtr_wrong = Xtr[:, (ytr_est != ytr.flatten()).flatten()]

if should_plot:
    plt.figure()
    plt.title('Estimated class labels')
    plt.scatter(Xtr_c0[0,:], Xtr_c0[1,:], c='r')
    plt.scatter(Xtr_c1[0,:], Xtr_c1[1,:],  c='b')
    #plt.scatter(Xtr_wrong[0,:], Xtr_wrong[1,:], c='r', s=40)

# to diagnose misclassification on the training set
for i in range(0, len(ps_a)):
    if i < len(ps_a) - 1:
        pair = ps_a[i:i+2, :]
    else:
        pair = np.array([ps_a[i, :], ps_a[0, :]])

    if should_plot:
        plt.plot(pair[:,0], pair[:,1], 'g-')

for i in range(0, len(ps_b)):
    if i < len(ps_b) - 1:
        pair = ps_b[i:i+2, :]
    else:
        pair = np.array([ps_b[i, :], ps_b[0, :]])
    
    if should_plot:
        plt.plot(pair[:,0], pair[:,1], 'g-')

if should_plot:
    plt.show()

# TODO visualize where the errors are?
print('Training set accuracy: ' + str(np.sum(ytr_est == ytr.transpose()) / len(ytr)))
print(ytr.shape)
print(ytr_est.shape)
print(ytr[(ytr_est != ytr.transpose()).flatten()])
print(ytr_est[(ytr_est != ytr.transpose()).flatten()])
print('Test set accuracy: ' + str(np.sum(yts_est == yts.transpose()) / len(yts)))


"""
Problem 2
"""

def logistic(W, X):
    #print(W.shape)
    #print(X.shape)
    return (1 / (1 + np.exp((-1) * np.dot(W, X.transpose())))).transpose()

def logistic_regression(X, y):
    '''
    # only for current version of cross-entropy cost function
    y = np.copy(y)
    y[y == 0] = -1
    '''

    print('Fitting logistic regression parameters...')
    # add the column of ones for the bias weights
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    print(X)

    # initial parameter guesses
    # only free parameters are the bias and input component weights
    W = np.zeros((1, X.shape[1]))
    #N = X.shape[1]
    # schedule?
    rate = 0.5

    last_error = np.nan

    N = X.shape[0]
    print('N=' + str(N))

    # fit the parameters using gradient descent
    while True:
        y_est = logistic(W, X)
        # TODO simplify?
        error = (1/N) * np.sum((-y)*np.log(y_est) - np.log(1-y_est)*(1-y))

        tmp = np.zeros(W.shape)

        # it didn't seem like this would have been any different (going component by component?)
        for j in range(0, X.shape[1]):
            for n in range(0, N):
                tmp[0,j] = tmp[0,j] + (y_est[n] - y[n]) * X[n,j]
        error_grad = (1/N) * tmp

        '''
        # TODO vectorize
        # "cross-entropy" cost function
        # TODO is this cost func convex?
        # note: this cost function relies on y being in {-1,1}
        error = 0
        for n in range(0, N):
            error = error + np.log(1 + np.exp((-y[n]) * np.dot(W, X[n,:].transpose())))
            #print(np.max(np.dot(W, X[n,:].transpose())))
            #print(np.min(np.dot(W, X[n,:].transpose())))
        error = (1/N) * error
        #print('cross ent error=' + str(error))

        # standard gradient descent update
        tmp = np.zeros((1,3))
        for n in range(0, N):
            # TODO first X slice transposed correctly?
            # note the lack of the negative sign in the exp term
            tmp = tmp + (-y[n] * X[n,:]) / \
                    (1 + np.exp(y[n] * np.dot(W, X[n,:].transpose())))
            #print(np.dot(W, X[n,:]).shape)
        error_grad = tmp * (1/N)

        # stochastic gradient descent update
        n = np.random.randint(0, N)
        error_grad = (-y[n] * X[n,:]) / (1 + np.exp(y[n] * np.dot(W, X[n,:].transpose())))
        '''

        print('error=' + str(error))
        #print(error_grad)
        print('accuracy=' + str(accuracy(y_est, y)))


        #print(y_est[:20])
        #print(y[:20])
        #break
    
        W = W - rate * error_grad
        
        #if np.isclose(last_error, error):
        if np.isnan(error) or last_error == error:
            break
        last_error = error

    return W

def accuracy(y_est, y):
    return np.sum(np.round(y_est) == y) / np.size(y)

# 2.3

B = scipy.io.loadmat('datasetB.mat')
# TODO this may be transposed
Xtr = B['Xtr']
ytr = B['ytr']
Xts = B['Xts']
yts = B['yts']

W = logistic_regression(Xtr, ytr)

if should_plot:
    plot_by_class(Xtr, ytr)

# TODO should not be 3x3
#print(W.shape)

# evaluates the decision boundary at x2 to find the corresponding x1
bound_x2y = lambda x2: (-W[0,0] - W[0,2]*x2) / W[0,1]
bound_y2x = lambda x1: (-W[0,0] - W[0,1]*x1) / W[0,2]

# now plot the learned decision boundary
# decision boundary: x1 = (-w0 - w2*x2) / w1
ax = plt.gca()
prev_x = ax.get_xlim()
prev_y = ax.get_ylim()
x = prev_x[0]
y = prev_y[0]
p_0 = (x, bound_x2y(x))
p_1 = (bound_y2x(y), y)

if should_plot:
    plt.plot(p_0, p_1, 'g-')
    ax.set_xlim(prev_x)
    ax.set_ylim(prev_y)
    plt.show()

'''
C = scipy.io.loadmat('datasetC.mat')
Xtr = C['Xtr']
ytr = C['ytr']
Xts = C['Xts']
yts = C['yts']
'''

# TODO remap features in non linearly separable C
