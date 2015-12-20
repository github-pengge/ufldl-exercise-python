# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import sys
sys.path.append('../selfTaughtLearning')
sys.path.append('../sparseAutoEncoder')
import gradient
import mnist
import numpy as np
import scipy.optimize
import time
import pickle

class softmax:
    def __init__(self,num_classes,input_size,lamda=1e-4,options=None):
        self.num_classes    = num_classes
        self.input_size     = input_size
        self.lamda          = lamda
        self.options        = options
        self.opt_theta      = None
        if(not self.options):
            self.options    = dict()
        # if(self.options.get('method',False) == False):
        #     self.options['method']  = 'L-BFGS-B'
        if(self.options.get('disp','_UNSET_') == '_UNSET_'):
            self.options['disp'] = True
        if(self.options.get('maxiter',False) == False):
            self.options['maxiter'] = 100

    @staticmethod
    def data_prepare(data,label):
        # Currently, there is no more we can do
        return data,label

    def softmax_cost(self, theta, data, labels):
        m           = data.shape[1]
        penalty     = 0.5 * self.lamda * np.sum(theta ** 2)
        theta       = np.reshape(theta,(self.num_classes,self.input_size))
        M           = theta.dot(data)
        M           = M - M.max(axis=0)
        exp_term    = np.exp(M) / np.exp(M).sum(axis=0) # size: c*m (c: num of classes)
        indicator   = np.zeros([self.num_classes, m])
        indicator[labels.flatten(),np.arange(m)] = 1
        ind_vec     = indicator.reshape((1,-1))
        exp_term_vec= exp_term.reshape((-1,1))
        loss        = -1.0 / m * ind_vec.dot(np.log(exp_term_vec))
        cost        = loss + penalty
        grad        = -1.0 / m * (indicator - exp_term).dot(data.T)\
                      + self.lamda * theta

        grad        = grad.flatten()

        del exp_term, indicator, theta, M, ind_vec, exp_term_vec
        return (cost, grad)

    def check_gradient(self,data,labels):
        theta           = 0.005 * np.random.random([self.num_classes * self.input_size])
        J               = lambda theta: self.softmax_cost(theta,data,labels)
        cost,grad       = self.softmax_cost(theta,data,labels)
        num_grad        = gradient.compute_gradient(J,theta)
        diff            = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
        print('Diff: %s' % diff)
        print('The diff above should be very small.')

    def train(self, data, labels,fileName=None):
        theta           = 0.005 * np.random.random([self.num_classes * self.input_size])
        J               = lambda theta: self.softmax_cost(theta,data,labels)
        opt_result      = scipy.optimize.minimize(J,theta,method='L-BFGS-B',
                                                  jac=True,options=self.options)
        # self.opt_theta,_,_ = scipy.optimize.fmin_l_bfgs_b(J,theta,
        #                                                   maxiter=self.options['maxiter'],
        #                                                   iprint=20,m=20)
        self.opt_theta  = opt_result.x

        if(fileName == None):
            fileName    = 'softmax_opt_theta.pkl'
        file            = open(fileName,'wb')
        pickle.dump(self.opt_theta,file)
        file.close()

        # print(opt_result)

    def predict(self,data):
        theta       = self.opt_theta.reshape(self.num_classes,self.input_size)
        prob        = theta.dot(data) # Actually, prob proportionals to theta.dot(data)
        predict     = np.argmax(prob,axis=0)
        return predict

    @staticmethod
    def static_predict(data,opt_theta,num_classes,input_size):
        theta       = opt_theta.reshape(num_classes,input_size)
        prob        = theta.dot(data) # Actually, prob proportionals to theta.dot(data)
        predict     = np.argmax(prob,axis=0)
        return predict

if __name__ == '__main__':
    begin           = time.time()
    images          = mnist.load_images('../data/mnist/train-images.idx3-ubyte')
    labels          = mnist.load_labels('../data/mnist/train-labels.idx1-ubyte')
    # print(images.shape,labels.shape)

    # if you want to run on a computer with weak computation power, use following two line
    # images          = images[0:400,]
    # labels          = labels[:]

    m               = images.shape[1]
    num_classes     = 10
    input_size      = 28 * 28
    lamda           = 1e-4
    softmax_model   = softmax(num_classes,input_size,lamda)

    # images, labels  = softmax_model.data_prepare(images,labels)

    # check gradient
    # softmax_model.check_gradient(images,labels)

    # train softmax model
    softmax_model.train(images,labels)

    # softmax classifier predict
    predict_labels  = softmax_model.predict(images)

    # compute correct rate
    correct_rate    = (predict_labels == labels).mean()

    end             = time.time()
    print('Softmax correct rate: %s.' % (correct_rate))
    print('Elapsed time: %s secs.' % (end - begin))
