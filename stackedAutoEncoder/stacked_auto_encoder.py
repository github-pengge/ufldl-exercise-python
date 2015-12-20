# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import sys
sys.path.append('../sparseAutoEncoder')
sys.path.append('../selfTaughtLearning')
sys.path.append('../softmax')
import numpy as np
import softmax
import mnist
import train
import sparse_auto_encoder
import stl
import gradient
import scipy.optimize
import pickle

class stacked_auto_encoder:
    def __init__(self,input_size,num_classes,hidden_sizeL1,hidden_sizeL2,
                 sparsity_param,lamda,beta):
        self.input_size     = input_size
        self.num_classes    = num_classes
        self.hidden_sizeL1  = hidden_sizeL1
        self.hidden_sizeL2  = hidden_sizeL2
        self.sparsity_param = sparsity_param
        self.lamda          = lamda
        self.beta           = beta

    def cost(self,theta,hidden_size,data,labels):
        '''
        cost: Takes a trained softmax_heta and a training data set
        with labels, and returns cost and gradient using a stacked autoencoder
        model. Used for finetuning.
        :param theta:       trained weights from the auto encoder
        :param hidden_size: the number of hidden units *at the 2nd layer*
        :param data:        Our matrix containing the training data as columns.
                            So, data(:,i) is the i-th training example.
        :param labels:      A vector containing labels, where labels(i) is the
                            label for the i-th training example
        :return: (cost,grad) in tuple
        '''

        # unroll softmax model
        softmax_theta       = np.reshape(theta[0:hidden_size * self.num_classes],
                                         newshape=(self.num_classes,hidden_size))
        stack               = self.params2stack(theta[hidden_size * self.num_classes:],
                                                self.net_config)

        # compute softmax theta geadient
        m                   = data.shape[1] # num of data
        a                   = [data]
        z                   = [0]
        for layer in range(len(stack)):
            z.append(stack[layer]['w'].dot(a[layer]) + np.atleast_2d(stack[layer]['b']).T)
            a.append(sparse_auto_encoder.activation(z[layer + 1]))

        M                   = softmax_theta.dot(a[-1])
        M                   = M - M.max(axis=0)
        exp_term            = np.exp(M) / np.exp(M).sum(axis=0)
        indicator           = np.zeros([self.num_classes,m])
        indicator[labels.flatten(),np.arange(m)] = 1
        ind_vec     = indicator.reshape((1,-1))
        exp_term_vec= exp_term.reshape((-1,1))
        penalty             = 0.5 * self.lamda * np.sum(softmax_theta ** 2)
        loss                = -1.0 / m * ind_vec.dot(np.log(exp_term_vec))
        cost                = loss + penalty
        softmax_theta_grad  = -1.0 / m * (indicator - exp_term).dot(a[-1].T) + \
                                self.lamda * softmax_theta
        stack_grad          = np.empty((len(stack)),dtype=object)
        delta               = np.empty((len(stack) + 1),dtype=object)
        delta[-1]           = -softmax_theta.T.dot(indicator - exp_term) * \
                                sparse_auto_encoder.activation_prime(z[-1])
        for i in range(len(stack)):
            layer           = len(stack) - i - 1
            delta[layer]    = stack[layer]['w'].T.dot(delta[layer + 1]) * \
                                sparse_auto_encoder.activation_prime(z[layer])
            stack_grad[layer]       = dict()
            stack_grad[layer]['w']  = 1.0 / m * delta[layer + 1].dot(a[layer].T) # here, we don't have to consider penalty
            stack_grad[layer]['b']  = 1.0 / m * np.sum(delta[layer + 1],axis=1)

        grad                = np.concatenate((softmax_theta_grad.flatten(),
                                              self.stack2params(stack_grad)[0].flatten()),
                                             axis=0)
        del exp_term, indicator, theta, M, ind_vec, exp_term_vec, z, a
        return cost,grad

    def params2stack(self,stack_param,net_config):
        '''
        Converts a flattened parameter vector into a nice "stack" structure
        for us to work with. This is useful when you're building multilayer
        networks.
        Usage:
            stack = params2stack(params, net_config)

        params - flattened parameter vector
        net_config - auxiliary variable containing
                    the configuration of the network

        '''
        stack   = []
        begin   = 0
        for i in range(len(net_config['layer_sizes'])):
            stack.append(dict())
            if(i == 0):
                end = begin + net_config['layer_sizes'][i] * net_config['input_size']
                stack[i]['w'] = np.reshape(stack_param[begin:end],
                                       newshape=(net_config['layer_sizes'][i],net_config['input_size']))
            else:
                end = begin + net_config['layer_sizes'][i] * net_config['layer_sizes'][i - 1]
                stack[i]['w'] = np.reshape(stack_param[begin:end],
                                       newshape=(net_config['layer_sizes'][i],net_config['layer_sizes'][i - 1]))
            begin   = end
            end     = begin + net_config['layer_sizes'][i]
            stack[i]['b'] = np.reshape(stack_param[begin:end],
                                       newshape=(net_config['layer_sizes'][i]))
            begin = end

        return stack

    def stack2params(self,stack):
        '''
        Converts a "stack" structure into a flattened parameter vector and also
        stores the network configuration. This is useful when working with
        optimization toolboxes such as minFunc.
        Usage:
            params, net_config = stack2params(stack)

        stack - the stack structure, where stack[0]['w'] = weights of first layer
                                           stack[0]['b] = weights of first layer
                                            stack[1]['w'] = weights of second layer
                                           stack[1]['b'] = weights of second layer
                                            ... etc.
        '''
        params                      = []
        net_config                  = dict()
        net_config['layer_sizes']   = []
        if(len(stack) == 0):
            net_config['input_size'] = 0
        else:
            net_config['input_size'] = stack[0]['w'].shape[1]

        for i in range(len(stack)):
            params.extend(stack[i]['w'].flatten())
            params.extend(stack[i]['b'].flatten())
            assert stack[i]['w'].shape[0] == stack[i]['b'].shape[0],\
                'The bias should be a %s vector of x1' % (stack[i]['w'].shape[0])
            if(i < len(stack) - 1):
                assert stack[i]['w'].shape[0] == stack[i + 1]['w'].shape[1],\
                'The adjacent layers L%s and L%s should have matching sizes.' % (i + 1, i + 2)
            net_config['layer_sizes'].append(stack[i]['w'].shape[0])

        return np.array(params),net_config

    def train(self,data,labels,maxiter=400): # 对应stackedAEExercise.m
        # train the first sparse auto encoder
        sae1_opt_theta  = train.train(data,self.input_size,self.hidden_sizeL1,
                                    self.sparsity_param,self.lamda,self.beta,
                                      max_iter=maxiter)

        # train the second sparse auto encoder
        # encode using the first sparse auto encoder
        sae1_feature    = stl.feed_forward_auto_encoder(sae1_opt_theta,
                                                        self.hidden_sizeL1,
                                                        self.input_size,data)
        sae2_opt_theta  = train.train(sae1_feature,self.hidden_sizeL1,
                                    self.hidden_sizeL2,self.sparsity_param,
                                    self.lamda,self.beta,max_iter=maxiter)

        # train the softmax classifier
        sae2_feature    = stl.feed_forward_auto_encoder(sae2_opt_theta,
                                                       self.hidden_sizeL2,
                                                       self.hidden_sizeL1,
                                                       sae1_feature)
        softmax_model   = softmax.softmax(self.num_classes,self.hidden_sizeL2,
                                          self.lamda)
        softmax_model.train(sae2_feature,labels)

        # finetune softmax model
        # Initialize parameters
        basic_size1     = self.hidden_sizeL1 * self.input_size
        basic_size2     = self.hidden_sizeL2 * self.hidden_sizeL1
        stack           = np.empty([2],dtype=object)
        stack[0]        = dict()
        stack[1]        = dict()
        stack[0]['w']   = np.reshape(sae1_opt_theta[0:basic_size1],
                                     newshape=(self.hidden_sizeL1,self.input_size))
        stack[0]['b']   = sae1_opt_theta[2 * basic_size1:2 * basic_size1 + self.hidden_sizeL1]
        stack[1]['w']   = np.reshape(sae2_opt_theta[0:basic_size2],
                                     newshape=(self.hidden_sizeL2,self.hidden_sizeL1))
        stack[1]['b']   = sae2_opt_theta[2 * basic_size2:2* basic_size2 + self.hidden_sizeL2]
        stack_param, self.net_config \
                        = self.stack2params(stack)
        sae_theta       = np.concatenate((softmax_model.opt_theta,stack_param),axis=0)

        # finetune (train)
        J               = lambda sae_theta:self.cost(sae_theta,self.hidden_sizeL2,
                                                     data,labels)
        self.opt_sae_theta,_,_ = \
            scipy.optimize.fmin_l_bfgs_b(J,sae_theta,iprint=25,maxiter=maxiter,disp=True)

        file            = open('opt_sae_theta.pkl','wb')
        pickle.dump(self.opt_sae_theta,file)
        file.close()

    def predict(self,data):
        softmax_theta   = np.reshape(self.opt_sae_theta[0:self.hidden_sizeL2 * self.num_classes],
                                     newshape=(self.num_classes,self.hidden_sizeL2))
        stack           = self.params2stack(self.opt_sae_theta[self.hidden_sizeL2 * self.num_classes:],
                                                self.net_config)
        # encode
        a                   = [data]
        z                   = [0]
        for layer in range(len(stack)):
            z.append(stack[layer]['w'].dot(a[layer]) + np.atleast_2d(stack[layer]['b']).T)
            a.append(sparse_auto_encoder.activation(z[layer + 1]))

        # softmax predict
        M                   = softmax_theta.dot(a[-1])
        predict_label       = np.argmax(M,axis=0)
        return predict_label

if __name__ == '__main__':
    DEBUG   = False
    images  = mnist.load_images('../data/mnist/train-images.idx3-ubyte')
    labels  = mnist.load_labels('../data/mnist/train-labels.idx1-ubyte')
    input_size      = 28 * 28
    num_classes     = 10
    hidden_sizeL1   = 200    # Layer 1 Hidden Size
    hidden_sizeL2   = 200    # Layer 2 Hidden Size
    sparsity_param  = 0.1    # desired average activation of the hidden units.
                             # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                             #  in the lecture notes).
    lamda           = 3e-3   # weight decay parameter
    beta            = 3      # weight of sparsity penalty term

    # images = images[:, :100]
    # labels = labels[:100]
    # # only top input_size most-varying input elements (pixels)
    # indices = images.var(1).argsort()[-input_size:]
    # images = images[indices, :]


    import time
    begin           = time.time()
    sae             = stacked_auto_encoder(input_size,num_classes,hidden_sizeL1,
                                           hidden_sizeL2,sparsity_param,lamda,beta)
    sae.train(images,labels,maxiter=400)
    predict_label   = sae.predict(images)
    correct_rate    = (predict_label == labels).mean()
    end             = time.time()
    print('Elapsed time: %s' % (end - begin))
    print('Correct_rate: %s' % (correct_rate))


    # check gradient
    if(DEBUG):
        inputSize = 4
        hiddenSize = 5
        lamda = 0.01
        data   = np.random.random((inputSize, 5))
        labels = np.array([1 ,2 ,1 ,2 ,1]) - 1
        numClasses = 2
        sae             = stacked_auto_encoder(inputSize,numClasses,hiddenSize,
                                               hiddenSize,sparsity_param,lamda,beta)

        stack = np.empty((2),dtype=object)
        stack[0]        = dict()
        stack[1]        = dict()
        stack[0]['w'] = 0.1 * np.random.random((3, inputSize))
        stack[0]['b'] = np.zeros((3))
        stack[1]['w'] = 0.1 * np.random.random((hiddenSize, 3))
        stack[1]['b'] = np.zeros((hiddenSize))
        softmaxTheta  = 0.005 * np.random.random((hiddenSize * numClasses))

        [stack_params, net_config] = sae.stack2params(stack)
        stackedAETheta      = np.concatenate((softmaxTheta,stack_params),axis=0)
        sae.net_config      = net_config
        sae.opt_sae_theta   = stackedAETheta

        [cost, grad]    = sae.cost(stackedAETheta,hiddenSize,data,labels)
        J               = lambda theta:sae.cost(theta,hiddenSize,data,labels)
        num_grad        = gradient.compute_gradient(J,sae.opt_sae_theta)
        print(np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad))