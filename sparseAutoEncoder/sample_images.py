# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'
import scipy.io
import numpy as np

def normalize_data(images,axis=1):
    mean_img = np.mean(images,axis=axis,keepdims=True)
    images = images - mean_img

    # Truncate to +/- 3 standard deviations and scale to -1 and +1
    pstd = 3 * images.std()
    images = np.maximum(np.minimum(images, pstd), -pstd) / pstd

    # Rescale from [-1,+1] to [0.1,0.9]
    images = (images + 1) * 0.4 + 0.1
    return images

def sample_images(mat_file,data_name,patch_size,num_patches):
    image_data = scipy.io.loadmat(mat_file)[data_name]
    image_size = image_data.shape[0]
    num_images = image_data.shape[2]
    patches = np.zeros([patch_size*patch_size,num_patches])

    for i in range(num_patches):
        image_id = np.random.randint(0,num_images-1)
        image_x = np.random.randint(0,image_size-patch_size)
        image_y = np.random.randint(0,image_size-patch_size)
        patches[:,i] = image_data[image_x:(image_x+patch_size),
                       image_y:(image_y+patch_size),
                       image_id].reshape(patch_size*patch_size)
    return normalize_data(patches)

if __name__ == '__main__':
    patches = sample_images('../data/IMAGES.mat','IMAGES',
                            patch_size = 8,num_patches = 10000)
    print(patches.shape)