# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np

act_fun = 'sigmoid'
def activation(x,fun=act_fun):
    act_dict = {'sigmoid':sigmoid,'tanh':tanh,'linear':linear}
    return act_dict[fun](x)

def activation_prime(x,fun=act_fun):
    act_prime_dict = {'sigmoid':sigmoid_prime,'tanh':tanh_prime,'linear':linear_prime}
    return act_prime_dict[fun](x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return (1.0/(1 + np.exp(-x)))

def linear(x):
    return x

def tanh_prime(x):
    return (1 - tanh(x) ** 2)

def sigmoid_prime(x):
    y = sigmoid(x)
    return (y * (1-y))

def linear_prime(x):
    return 1

def initialize(hidden_size,visible_size):
    # Initialize parameters randomly based on layer sizes.
    # we'll choose weights uniformly from the interval [-r, r]
    r  = np.sqrt(6) / np.sqrt(hidden_size+visible_size+1)
    W1 = np.random.rand(hidden_size, visible_size) * 2 * r - r
    W2 = np.random.rand(visible_size, hidden_size) * 2 * r - r

    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(visible_size, dtype=np.float64)

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used with minFunc.
    theta = np.concatenate((W1.flatten(),
                            W2.flatten(),
                            b1,
                            b2))
    return theta

def sparse_auto_encoder_cost(theta,hidden_size,visible_size,
                          lamda,sparsity_param,beta,data):
    '''
    :param theta: all the params of the auto encoder, including weights and b
    :param hidden_size: the number of hidden units (probably 25)
    :param visible_size: the number of visible units (probably 64)
    :param lamda: weight decay parameter
    :param sparsity_param: the desire average activation for hidden units
    :param beta: weight of sparsity penalty term
    :param data: training data, data[:,i] is the i-th training image
    :return: (cost, gradient) tuple

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    '''
    sp          = sparsity_param
    basic_size  = hidden_size * visible_size
    W1          = np.reshape(theta[0: basic_size], (hidden_size,visible_size))
    W2          = np.reshape(theta[basic_size: 2 * basic_size], (visible_size,hidden_size))
    b1          = np.reshape(theta[2 * basic_size: 2 * basic_size + hidden_size],(hidden_size,1))
    b2          = np.reshape(theta[2 * basic_size + hidden_size:],(visible_size,1))

    # Cost and gradient variables
    # Here, we initialize them to zeros.
    m                   = data.shape[1] # num of data
    hidden_layer_input  = W1.dot(data) + b1 # size: h*d
    hidden_layer        = activation(hidden_layer_input)
    output_layer_input  = W2.dot(hidden_layer) + b2 # size: v*m
    output_layer        = activation(output_layer_input)
    rho                 = 1.0 / m * np.sum(hidden_layer,axis=1) # size: h
    diff                = output_layer - data
    delta2              = diff * activation_prime(output_layer_input) # size: v*m
    delta1              = (W2.T.dot(delta2) + np.atleast_2d(beta * (-sp / rho + (1 - sp) / (1 - rho))).T) \
                            * activation_prime(hidden_layer_input) # size: h*m
    delta_W2            = 1.0 / m * delta2.dot(hidden_layer.T) + lamda * W2# size: v*h
    delta_b2            = 1.0 / m * np.atleast_2d(delta2.sum(axis=1)) # size: 1*v
    delta_W1            = 1.0 / m * delta1.dot(data.T) + lamda * W1 # size: h*v
    delta_b1            = 1.0 / m * np.atleast_2d(delta1.sum(axis=1)) # size: 1*h

    square_error        = 0.5 * 1.0 / m * np.sum(diff ** 2)
    penalty_term        = 0.5 * lamda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    KL_div              = beta * np.sum(sp * np.log(sp / rho) +
                                  (1 - sp) * np.log((1 - sp) / (1 - rho)))
    cost                = square_error + penalty_term + KL_div
    theta_grad          = np.concatenate((delta_W1.flatten(),
                                        delta_W2.flatten(),
                                        delta_b1.flatten(),
                                        delta_b2.flatten()))
    del hidden_layer_input,hidden_layer,output_layer_input,output_layer,rho,diff
    del delta2,delta1,W1,W2,b1,b2
    return (cost, theta_grad)


