# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import sparse_auto_encoder as SAE
import sample_images as sample
import display_network as display
import scipy.optimize

def train(patches, visible_size=64, hidden_size=25, sparsity_param=0.01,
          lamda=1e-4, beta=3,max_iter=400):
    theta   = SAE.initialize(hidden_size,visible_size)
    J       = lambda theta: SAE.sparse_auto_encoder_cost(theta,hidden_size,visible_size,
                          lamda,sparsity_param,beta,patches)
    options = {'maxiter':max_iter,'disp':True}
    opt_result  = scipy.optimize.minimize(J,theta,method='L-BFGS-B',
                                      jac=True,options=options)
    opt_theta = opt_result.x
    print(opt_result)

    return opt_theta

if __name__ == '__main__':
    visible_size    = 64
    hidden_size     = 25
    sparsity_param  = 0.01
    lamda           = 1e-4
    beta            = 3
    max_iter        = 400
    patches         = sample.sample_images('../data/IMAGES.mat','IMAGES',
                                            patch_size = 8,num_patches = 10000)
    theta           = train(patches,visible_size,hidden_size,sparsity_param,
                            lamda,beta,max_iter)
    display.display_network(theta[0:hidden_size*visible_size].reshape(
                            hidden_size,visible_size).T)
