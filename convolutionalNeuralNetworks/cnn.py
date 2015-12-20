# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np
import scipy.signal
import time
import sys
sys.path.append('../selfTaughtLearning')
sys.path.append('../sparseAutoEncoder')
sys.path.append('../softmax')
import stl
import pickle
import scipy.io
from sparse_auto_encoder import activation
import display_network
import softmax

def cnn_convolve(patch_dim, num_features, images, W, b, ZCA_White, mean_patch):
    image_dim, _ ,image_channels, num_images = images.shape

    WT                  = W.dot(ZCA_White)
    b_WT_mean           = b - WT.dot(np.squeeze(mean_patch))

    convolved_features  = np.zeros([num_features, num_images, image_dim - patch_dim + 1, image_dim - patch_dim + 1])
    feature_size        = patch_dim * patch_dim
    for image_num in range(num_images):
        for feature_num in range(num_features):
            convolved_image     = np.zeros((image_dim - patch_dim + 1, image_dim - patch_dim + 1))
            for channel in range(image_channels):
                feature         = np.reshape(WT[feature_num,channel * feature_size: (channel + 1) * feature_size],
                                            newshape=(patch_dim,patch_dim))
                feature         = np.flipud(np.fliplr(np.squeeze(feature)))
                im              = np.squeeze(images[:,:,channel,image_num])

                # Convolve image with feature,
                # then, subtract the bias unit (correcting for the mean subtraction as well),
                # after that, apply the sigmoid function to get the hidden activation
                # The convolved feature is the sum of the convolved values for all channels
                convolved_image += scipy.signal.convolve2d(im,feature,'valid')

            convolved_image     += b_WT_mean[feature_num]
            convolved_features[feature_num,image_num,:,:]  = activation(convolved_image)
    return convolved_features

def cnn_pool(pool_dim, convolved_features,method='mean-nonOverlapping'):
    # todo: method='mean-continuous' has just pass the test, but we haven't check other methods yet.
    num_features,num_images,convolved_dim,_ = convolved_features.shape
    pool_method             = method.lower().split('-')
    if('nonOverlapping' in pool_method):
        pool_feature_dim    = int(convolved_dim / np.float(pool_dim))
    elif('invariant' in pool_method):
        pool_feature_dim    = convolved_dim - pool_dim + 1
    else:
        raise ValueError('No such method: %s.' % method)
    if('mean' in pool_method):
        tricky              = np.mean
    elif('max' in pool_method):
        tricky              = np.max
    else:
        raise ValueError('No such method: %s.' % method)

    pooled_features     = np.zeros([num_features,num_images,pool_feature_dim,pool_feature_dim])

    for image_num in range(num_images):
        for feature_num in range(num_features):
            for block in range(pool_feature_dim ** 2):
                pos = [block // pool_feature_dim, block % pool_feature_dim]
                if('nonOverlapping' in pool_method):
                    fb  = [pos[i] * pool_dim for i in range(len(pos))]
                elif('invariant' in pool_method):
                    fb  = pos
                pooled_features[feature_num,image_num,pos[0],pos[1]] = \
                        tricky(convolved_features[feature_num, image_num,
                                fb[0]:fb[0] + pool_dim, fb[1]:fb[1] + pool_dim])

    return pooled_features



if __name__ == '__main__':
    DEBUG           = False
    use_pre_trained = True
    load_W_b        = False

    image_dim       = 64    # image dimension
    image_channels  = 3     # number of channels (rgb, so 3)
    patch_dim       = 8     # patch dimension
    num_patches     = 50000 # number of patches
    visible_size    = patch_dim * patch_dim * image_channels  # number of input units
    output_size     = visible_size  # number of output units
    hidden_size     = 400           # number of hidden units
    epsilon         = 0.1	        # epsilon for ZCA whitening
    pool_dim        = 19            # dimension of pooling region

    # load feature learned from linear decoder
    if(load_W_b):
        file            = open('../linearDecoder/opt_theta.pkl','rb')
        theta           = pickle.load(file,encoding='latin1')
        file.close()
        file            = open('../linearDecoder/ZCA_White.pkl','rb')
        ZCA_White       = pickle.load(file,encoding='latin1')
        file.close()
        file            = open('../linearDecoder/mean_image.pkl','rb')
        mean_image      = pickle.load(file,encoding='latin1')
        file.close()

        W               = np.reshape(theta[0:visible_size * hidden_size],
                                    newshape=(hidden_size, visible_size))
        b               = theta[2 * hidden_size * visible_size:
                                2 * hidden_size * visible_size + hidden_size]

        display_network.display_network(W.dot(ZCA_White).T, gray_color=False)

    # load training data
    if(not use_pre_trained):
        data            = scipy.io.loadmat('../data/stlTrainSubset.mat')
        train_images    = data['trainImages']
        num_train_images= data['numTrainImages'][0][0]
        train_labels    = data['trainLabels'] - 1 # label index begins from 0

    # load testing data
    test_data       = scipy.io.loadmat('../data/stlTestSubset.mat')
    test_images     = test_data['testImages']
    num_test_images = test_data['numTestImages'][0][0]
    test_labels     = test_data['testLabels'] - 1 # label index begins from 0

    # Debug
    if(DEBUG):
        # check convolution
        conv_images         = train_images[:, :, :, 0:8]
        convolved_features  = cnn_convolve(patch_dim, hidden_size, conv_images, W, b, ZCA_White, mean_image)
        # For 1000 random points
        for i in range(1000):
            feature_num = np.random.randint(0, hidden_size)
            image_num   = np.random.randint(0, 8)
            image_row   = np.random.randint(0, image_dim - patch_dim + 1)
            image_col   = np.random.randint(0, image_dim - patch_dim + 1)

            patch = conv_images[image_row:image_row + patch_dim, image_col:image_col + patch_dim, :, image_num]
            patch = np.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
            patch = np.reshape(patch, (patch.size, 1))
            patch = patch - mean_image # we use keepdims = True when finding a mean image
            patch = ZCA_White.dot(patch)

            features = stl.feed_forward_auto_encoder(theta, hidden_size, visible_size, patch)

            if abs(features[feature_num, 0] - convolved_features[feature_num, image_num, image_row, image_col]) > 1e-9:
                print('Convolved feature does not match activation from auto encoder\n')
                print('Feature Number    : %s\n' % feature_num)
                print('Image Number      : %s\n' % image_num)
                print('Image Row         : %s\n' % image_row)
                print('Image Column      : %s\n' % image_col)
                print('Convolved feature : %s\n' % convolved_features[feature_num,image_num,image_row,image_col])
                print('Sparse AE feature : %s\n' % features[feature_num, 0])
                raise Exception('Convolved feature does not match activation from auto encoder')
        print('Congratulations! Your convolution code passed the test.')

        # check pooling
        pooled_features = cnn_pool(pool_dim, convolved_features)
        test_matrix     = np.reshape(range(64), (8, 8))
        expected_matrix = [[np.mean(test_matrix[0:4,0:4]),np.mean(test_matrix[0:4,4:8])],
                           [np.mean(test_matrix[4:8,0:4]),np.mean(test_matrix[4:8,4:8])]]
        test_matrix     = np.reshape(test_matrix,(1,1,8,8))
        pooled_features = np.squeeze(cnn_pool(4, test_matrix))

        if np.any(pooled_features != expected_matrix):
            print('Pooling incorrect')
            print('Expected % s' % expected_matrix)
            print('Got %s' % pooled_features)
        else:
            print('Congratulations! Your pooling code passed the test.')

    begin                   = time.time()
    # training
    if(not use_pre_trained):
        step_size             = 50
        assert(hidden_size % step_size == 0)
        pooled_features_train   = np.zeros([hidden_size, num_train_images,
                                        np.floor((image_dim - patch_dim + 1) / pool_dim),
                                        np.floor((image_dim - patch_dim + 1) / pool_dim)])
        pooled_features_test    = np.zeros([hidden_size, num_test_images,
                                        np.floor((image_dim - patch_dim + 1) / pool_dim),
                                        np.floor((image_dim - patch_dim + 1) / pool_dim)])

        for conv_part in range(int(hidden_size / step_size)):
            feature_start   = conv_part * step_size
            feature_end     = (conv_part + 1) * step_size

            print('Step %s: features %s to %s' % (conv_part + 1, feature_start + 1, feature_end))
            Wt              = W[feature_start:feature_end, :]
            bt              = b[feature_start:feature_end]

            print('Convolving and pooling train images')
            convolved_features_this = cnn_convolve(patch_dim, step_size,
                                                    train_images, Wt, bt, ZCA_White, mean_image)
            pooled_features_this    = cnn_pool(pool_dim, convolved_features_this)
            pooled_features_train[feature_start:feature_end, :, :, :] = pooled_features_this

            print('Part %s, current totally elapsed time: %s secs' % (conv_part + 1, time.time() - begin))
            del convolved_features_this, pooled_features_this

            print('Convolving and pooling test images')
            convolved_features_this = cnn_convolve(patch_dim, step_size,
                                                    test_images, Wt, bt, ZCA_White, mean_image)
            pooled_features_this    = cnn_pool(pool_dim, convolved_features_this)
            pooled_features_test[feature_start:feature_end, :, :, :] = pooled_features_this
            print('Part %s, current elapsed time: %s secs' % (conv_part + 1, time.time() - begin))

            del convolved_features_this, pooled_features_this

        # save convolved data
        pkl_file    = open('convolved_training_and_testing_data.pkl','wb')
        pickle.dump([pooled_features_train,pooled_features_test],pkl_file)
        pkl_file.close()
        print('Convolution totally elapsed time: %s secs.' % (time.time() - begin))

    # load convolved data
    if(use_pre_trained):
        pkl_file    = open('convolved_training_and_testing_data.pkl','rb')
        [pooled_features_train,pooled_features_test] = pickle.load(pkl_file,encoding='latin1')
        pkl_file.close()

    # Use the convolved and pooled features to train a softmax classifier
    # Setup parameters for softmax
    softmax_lamda       = 1e-4
    num_classes         = 4
    softmax_input_size  = pooled_features_train.size / num_train_images

    # Reshape the pooledFeatures to form an input vector for softmax
    softmax_X           = np.transpose(pooled_features_train, [0,2,3,1])
    softmax_X           = np.reshape(softmax_X,
                                    newshape=(softmax_input_size, num_train_images),order='F')
    softmax_Y           = np.squeeze(train_labels)

    options             = dict()
    options['maxiter']  = 200
    softmax_model       = softmax.softmax(num_classes,softmax_input_size,softmax_lamda,options)
    softmax_model.train(softmax_X,softmax_Y)

    # testing
    softmax_X           = np.transpose(pooled_features_test, [0,2,3,1])
    softmax_X           = np.reshape(softmax_X,
                                     newshape=(pooled_features_test.size / num_test_images, num_test_images),
                                     order='F')
    softmax_Y           = np.squeeze(test_labels)
    predict_Y           = np.squeeze(softmax_model.predict(softmax_X))
    correct_rate        = (predict_Y == softmax_Y).mean()
    end                 = time.time()
    print('Correct rate: %s' % correct_rate)
    print('Totally elapsed time: %s secs.' % (end - begin))