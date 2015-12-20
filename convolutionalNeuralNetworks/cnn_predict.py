# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np
import pickle
import cnn
import sys
sys.path.append('../softmax')
import softmax as sf
from PIL import Image
import cv2

def cnn_predict(images,W,b,ZCA_White,mean_image,softmax_theta,pooling_method,num_classes,input_size): # using a specific model
    if(len(images.shape) < 4):
        images            = images[:,:,:,np.newaxis]
    num_images            = images.shape[3]
    pooling_method        = pooling_method.lower()
    if('continuous' in pooling_method):
        pooled_size       = np.floor((image_dim - patch_dim + 1) / pool_dim)
    else:
        pooled_size       = image_dim - patch_dim + 1 - pool_dim + 1

    # convolving and pooling
    step_size             = 50
    assert(hidden_size % step_size == 0)
    pooled_features       = np.zeros([hidden_size, num_images,
                                    pooled_size,pooled_size])

    for conv_part in range(int(hidden_size / step_size)):
        feature_start   = conv_part * step_size
        feature_end     = (conv_part + 1) * step_size

        print('Step %s: features %s to %s' % (conv_part + 1, feature_start + 1, feature_end))
        Wt              = W[feature_start:feature_end, :]
        bt              = b[feature_start:feature_end]

        print('Convolving and pooling train images')
        convolved_features_this = cnn.cnn_convolve(patch_dim, step_size,
                                                images, Wt, bt, ZCA_White, mean_image)
        pooled_features_this    = cnn.cnn_pool(pool_dim, convolved_features_this,pooling_method)
        pooled_features[feature_start:feature_end, :, :, :] = pooled_features_this

        del convolved_features_this, pooled_features_this

    # softmax predict
    softmax_X           = np.transpose(pooled_features, [0,2,3,1])
    softmax_X           = np.reshape(softmax_X,
                                     newshape=(input_size, num_images),
                                     order='F')
    predict             = sf.softmax.static_predict(softmax_X,softmax_theta,
                                                    num_classes,input_size)
    return predict

def django_interface(img):
    image_dim       = 64    # image dimension
    image_channels  = 3     # number of channels (rgb, so 3)
    patch_dim       = 8     # patch dimension
    num_patches     = 50000 # number of patches
    visible_size    = patch_dim * patch_dim * image_channels  # number of input units
    output_size     = visible_size  # number of output units
    hidden_size     = 400           # number of hidden units
    epsilon         = 0.1	        # epsilon for ZCA whitening
    pool_dim        = 19            # dimension of pooling region
    num_classes     = 4
    label_dict      = dict()
    label_dict[0]   = 'airplane'
    label_dict[1]   = 'car'
    label_dict[2]   = 'cat'
    label_dict[3]   = 'dog'

    pool_method     = 'mean-continuous'

    if('continuous' in pool_method.lower()):
        pooled_size       = np.floor((image_dim - patch_dim + 1) / pool_dim)
    else:
        pooled_size       = image_dim - patch_dim + 1 - pool_dim + 1

    theta_file      = open('../linearDecoder/opt_theta.pkl','rb')
    if(sys.version_info.major > 2):
        theta       = pickle.load(theta_file,encoding='latin1')
    else:
        theta       = pickle.load(theta_file)
    theta_file.close()
    mean_image_file = open('../linearDecoder/mean_image.pkl','rb')
    if(sys.version_info.major > 2):
        mean_image  = pickle.load(mean_image_file,encoding='latin1')
    else:
        mean_image  = pickle.load(mean_image_file)
    mean_image_file.close()
    ZCA_White_file  = open('../linearDecoder/ZCA_White.pkl','rb')
    if(sys.version_info.major > 2):
        ZCA_White   = pickle.load(ZCA_White_file,encoding='latin1')
    else:
        ZCA_White   = pickle.load(ZCA_White_file)
    ZCA_White_file.close()

    W               = np.reshape(theta[0:visible_size * hidden_size],
                                newshape=(hidden_size, visible_size))
    b               = theta[2 * hidden_size * visible_size:
                            2 * hidden_size * visible_size + hidden_size]

    softmax_theta_file = open('softmax_opt_theta.pkl','rb')
    if(sys.version_info.major > 2):
        softmax_theta  = pickle.load(softmax_theta_file,encoding='latin1')
    else:
        softmax_theta  = pickle.load(softmax_theta_file)
    softmax_theta_file.close()

    softmax_input_size = hidden_size * pooled_size * pooled_size

    img                = img.resize((image_dim,image_dim))
    img                = img / 255.0

    predict            = cnn_predict(img,W,b,ZCA_White,mean_image,softmax_theta,pool_method,
                                     num_classes,softmax_input_size)
    return "May be it's a(n) %s." % label_dict[predict[0]]


if __name__ == '__main__':
    image_dim       = 64    # image dimension
    image_channels  = 3     # number of channels (rgb, so 3)
    patch_dim       = 8     # patch dimension
    num_patches     = 50000 # number of patches
    visible_size    = patch_dim * patch_dim * image_channels  # number of input units
    output_size     = visible_size  # number of output units
    hidden_size     = 400           # number of hidden units
    epsilon         = 0.1	        # epsilon for ZCA whitening
    pool_dim        = 19            # dimension of pooling region
    num_classes     = 4
    label_dict      = dict()
    label_dict[0]   = 'airplane'
    label_dict[1]   = 'car'
    label_dict[2]   = 'cat'
    label_dict[3]   = 'dog'

    pool_method     = 'mean-continuous'

    if('continuous' in pool_method.lower()):
        pooled_size       = np.floor((image_dim - patch_dim + 1) / pool_dim)
    else:
        pooled_size       = image_dim - patch_dim + 1 - pool_dim + 1

    theta_file      = open('../linearDecoder/opt_theta.pkl','rb')
    if(sys.version_info.major > 2):
        theta       = pickle.load(theta_file,encoding='latin1')
    else:
        theta       = pickle.load(theta_file)
    theta_file.close()
    mean_image_file = open('../linearDecoder/mean_image.pkl','rb')
    if(sys.version_info.major > 2):
        mean_image  = pickle.load(mean_image_file,encoding='latin1')
    else:
        mean_image  = pickle.load(mean_image_file)
    mean_image_file.close()
    ZCA_White_file  = open('../linearDecoder/ZCA_White.pkl','rb')
    if(sys.version_info.major > 2):
        ZCA_White   = pickle.load(ZCA_White_file,encoding='latin1')
    else:
        ZCA_White   = pickle.load(ZCA_White_file)
    ZCA_White_file.close()

    W               = np.reshape(theta[0:visible_size * hidden_size],
                                newshape=(hidden_size, visible_size))
    b               = theta[2 * hidden_size * visible_size:
                            2 * hidden_size * visible_size + hidden_size]

    softmax_theta_file = open('softmax_opt_theta.pkl','rb')
    if(sys.version_info.major > 2):
        softmax_theta  = pickle.load(softmax_theta_file,encoding='latin1')
    else:
        softmax_theta  = pickle.load(softmax_theta_file)
    softmax_theta_file.close()

    softmax_input_size = hidden_size * pooled_size * pooled_size

    # import scipy.io
    # data            = scipy.io.loadmat('../data/stlTestSubset.mat')
    # img             = data['testImages'][:,:,:,0]
    # print(data['testImages'][0,0:10,0,0])
    # cv2.imshow('img',data['trainImages'][:,:,:,0])
    #
    # cv2.waitKey(10000)

    # img                = cv2.imread('../data/testImages/cat111.jpg')
    # img                = cv2.resize(img,None,fx=image_dim/img.shape[1],fy=image_dim/img.shape[0])
    # # img                = img / np.max(np.max(img,axis=0),axis=0) # change to float type
    # img                = img / 255

    img                  = Image.open('../data/testImages/cat000.jpg')
    print(np.array(img).shape)
    img                  = img.resize((image_dim,image_dim))
    img                  = np.asarray(img) / 255.0

    print(img.shape)
    predict            = cnn_predict(img,W,b,ZCA_White,mean_image,softmax_theta,pool_method,
                                     num_classes,softmax_input_size)
    print(predict)
    print("May be it's a(n) %s, right?" % label_dict[predict[0]])
    # result: 0 for airplane, 1 for car, 2 for cat, 3 for dog