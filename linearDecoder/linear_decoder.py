# -*- coding : utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np
import scipy.io
import scipy.optimize
import pickle
import sys
sys.path.append('../sparseAutoEncoder')
import sparse_auto_encoder as SAE
import display_network as display

def cost(theta,visible_size,hidden_size,lamda,sparsity_param,beta,data):
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
    hidden_layer        = SAE.activation(hidden_layer_input)
    output_layer_input  = W2.dot(hidden_layer) + b2 # size: v*m
    output_layer        = SAE.activation(output_layer_input,'linear') # use a linear decoder
    rho                 = 1.0 / m * np.sum(hidden_layer,axis=1) # size: h
    diff                = output_layer - data
    delta2              = diff * SAE.activation_prime(output_layer_input,'linear') # size: v*m, use a linear decoder
    delta1              = (W2.T.dot(delta2) + np.atleast_2d(beta * (-sp / rho + (1 - sp) / (1 - rho))).T) \
                            * SAE.activation_prime(hidden_layer_input) # size: h*m
    delta_W2            = 1.0 / m * delta2.dot(hidden_layer.T) + lamda * W2 # size: v*h
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

def linear_decoder(data,visible_size,hidden_size,sparsity_param=0.035,
                   lamda=3e-3,beta=5,epsilon=0.1,options=None,disp_result=True):
    if(type(options) == type(None)):
        options             = dict()
    if(not options.get('method',False)):
        method              = 'L-BFGS-B'
    else:
        method              = options.pop('method')
    if(not options.get('maxiter',False)):
        options['maxiter']  = 400
    if(options.get('disp','_UNSET_') == '_UNSET_'):
        options['disp']     = True

    # apply ZCA Whitening on data
    mean_data       = np.mean(data,axis=1,keepdims=True)
    data            = data - mean_data
    sigma           = data.dot(data.T) / data.shape[1]
    u, s, _         = np.linalg.svd(sigma)
    ZCA_white       = u.dot(np.diag(1 / np.sqrt(s + epsilon)).T).dot(u.T)
    data            = ZCA_white.dot(data)
    # display.display_network(data[:,0:144],gray_color=False)


    # train the linear decoder on data
    theta           = SAE.initialize(hidden_size,visible_size)
    J               = lambda theta: cost(theta,visible_size,hidden_size,
                                         lamda,sparsity_param,beta,data)
    opt_result      = scipy.optimize.minimize(J,theta,method=method,jac=True,
                                              options=options)
    opt_theta       = opt_result.x

    file            = open('STL10_feature.pkl','wb')
    pickle.dump([opt_theta,ZCA_white,mean_data],file)
    file.close()

    if(disp_result):
        W           = np.reshape(opt_theta[0:visible_size * hidden_size],
                                 newshape=(hidden_size,visible_size))
        display.display_network(W.dot(ZCA_white).T,gray_color=False)
    return opt_theta

if __name__ == '__main__':
    patches     = scipy.io.loadmat('../data/stlSampledPatches.mat')['patches']

    image_channels  = 3        # number of channels (rgb, so 3)
    patch_dim       = 8        # patch dimension
    num_patches     = 100000   # number of patches
    visible_size    = patch_dim * patch_dim * image_channels  # number of input units
    output_size     = visible_size   # number of output units
    hidden_size     = 400           # number of hidden units
    sparsity_param  = 0.035         # desired average activation of the hidden units.
    lamda           = 3e-3          # weight decay parameter
    beta            = 5             # weight of sparsity penalty term
    epsilon         = 0.1	        # epsilon for ZCA whitening

    opt_theta       = linear_decoder(patches,visible_size,hidden_size,sparsity_param,
                                    lamda,beta,epsilon)
    print(opt_theta.shape)


    # check gradient
    # debugHiddenSize = 5
    # debugvisibleSize = 8
    # patches = np.random.random([8,10])
    # theta = SAE.initialize(debugHiddenSize, debugvisibleSize)
    # c,grad = cost(theta,debugvisibleSize,debugHiddenSize,lamda,sparsity_param,
    #               beta,patches)
    # ct   = lambda theta:cost(theta,debugvisibleSize,debugHiddenSize,lamda,
    #                          sparsity_param,beta,patches)
    # import gradient
    # num_grad = gradient.compute_gradient(ct,theta)
    # print(np.linalg.norm(grad - num_grad) / np.linalg.norm(grad + num_grad))