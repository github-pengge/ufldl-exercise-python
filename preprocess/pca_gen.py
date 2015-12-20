# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np
import sys
sys.path.append("..")
import sparseAutoEncoder.sample_images as sample
import sparseAutoEncoder.display_network as display
import pca_2d

def data_prepare(images):
    images -= np.mean(images,axis=0)
    return images

if __name__ == '__main__':
    patches             = sample.sample_images('../data/IMAGES_RAW.mat','IMAGESr',
                                patch_size=12,num_patches = 10000)
    patches             = data_prepare(patches)
    display.display_network(patches[:,0:20])

    U,S                 = pca_2d.pca(patches,percent=0.99)
    patches_rot         = U.T.dot(patches) # dimension reduction
    patches_whitening   = pca_2d.whitening(U,S,patches_rot,eps=0.1,method='ZCAWhitening')
    patches = U.dot(patches_rot) # recover images
    display.display_network(patches[:,0:20])
    # display.display_network(patches_whitening[:,0:20])

    # check covariance after PCAWhitening
    # print('covariance of patches_whitening: %s' %
    #       (patches_whitening.dot(patches_whitening.T)/patches_whitening.shape[1]))

