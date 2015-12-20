# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import mnist
import numpy as np
import sys
sys.path.append('../sparseAutoEncoder')
sys.path.append('../softmax')
from train import train
from sparse_auto_encoder import activation
import softmax as sm
import pickle
import time


def stl(train_data,train_label,unlabeled_data,input_size,num_labels,hidden_size,
        sparsity_param,lamda,beta,max_iter,theta_from_file=False):
    if(not theta_from_file):
        opt_theta       = train(unlabeled_data,input_size,hidden_size,
                                sparsity_param,lamda,beta,max_iter)
        file            = open('opt_theta.pkl','wb')
        pickle.dump(opt_theta,file) # fix_imports=True
        file.close()
    else:
        file            = open(theta_from_file,'rb')
        opt_theta       = pickle.load(file) # encoding='latin1'
        file.close()
    train_feature       = feed_forward_auto_encoder(opt_theta,hidden_size,
                                                    input_size,train_data)

    softmax_model       = sm.softmax(num_labels,hidden_size,lamda) # use a default options
    softmax_model.train(train_feature,train_label)

    return opt_theta, softmax_model

def classify(opt_theta,softmax_model,test_data,hidden_size,visible_size):
    test_feature        = feed_forward_auto_encoder(opt_theta,hidden_size,
                                                    visible_size,test_data)

    return softmax_model.predict(test_feature)

def feed_forward_auto_encoder(theta, hidden_size, visible_size, data):
    basic_size  = hidden_size * visible_size
    W1          = theta[0:basic_size].reshape(hidden_size,visible_size)
    b1          = theta[2 * basic_size: 2 * basic_size + hidden_size].reshape(hidden_size,1)
    feature     = activation(W1.dot(data) + b1)
    return feature

if __name__ == '__main__':
    begin           = time.time()
    images          = mnist.load_images('../data/mnist/train-images.idx3-ubyte')
    labels          = mnist.load_labels('../data/mnist/train-labels.idx1-ubyte')

    labeled_set     = np.nonzero((labels >= 0) * (labels <= 4))[0]
    unlabeled_set   = np.nonzero(labels >= 5)[0]
    num_train       = np.round(len(labeled_set) / 2)
    train_set       = labeled_set[0:num_train]
    test_set        = labeled_set[num_train:]

    unlabeled_data  = images[:,unlabeled_set]
    train_data      = images[:,train_set]
    train_label     = labels[train_set]

    test_data       = images[:,test_set]
    test_label      = labels[test_set]

    print('# examples in unlabeled set: %d' % (len(unlabeled_set)))
    print('# examples in supervised training set: %d' % (len(train_set)))
    print('# examples in supervised testing set: %d' % (len(test_set)))

    del images,labels
    input_size      = 28 * 28
    num_labels      = 5
    hidden_size     = 200
    sparsity_param  = 0.1
    lamda           = 3e-3
    beta            = 3
    max_iter        = 400
    opt_theta, softmax_model    = stl(train_data,train_label,unlabeled_data,
                                        input_size,num_labels,hidden_size,
                                        sparsity_param,lamda,beta,max_iter,
                                        theta_from_file=False)
    predict         = classify(opt_theta,softmax_model,test_data,hidden_size,input_size)
    correct_rate    = (predict == test_label).mean()
    end = time.time()
    print('Self taught learning, correct rate: %s.' % (correct_rate))
    print('Elapsed time: %s secs.' % (end - begin))