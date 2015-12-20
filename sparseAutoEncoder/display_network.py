# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'
import numpy as np
import matplotlib.pyplot as plt
import PIL

# def display_network(arr, title=None, show=True):
#     arr = arr - np.mean(arr)
#     L, M = arr.shape
#     sz = np.sqrt(L)
#     buf = 1
#
#     # Figure out pleasant grid dimensions
#     if M == np.floor(np.sqrt(M))**2:
#         n = m = np.sqrt(M)
#     else:
#         n = np.ceil(np.sqrt(M))
#         while (M%n) and n < 1.2*np.sqrt(M):
#             n += 1
#         m = np.ceil(M/n)
#
#     array = np.zeros([buf+m*(sz+buf), buf+n*(sz+buf)])
#
#     k = 0
#     for i in range(0, int(m)):
#         for j in range(0, int(n)):
#             if k>=M:
#                 continue
#             cmax = np.max(arr[:,k])
#             cmin = np.min(arr[:,k])
#             r = buf+i*(sz+buf)
#             c = buf+j*(sz+buf)
#             array[r:r+sz, c:c+sz] = (arr[:,k].reshape([sz,sz], order='F') - cmin) / (cmax-cmin)
#             k = k + 1
#     plt.figure()
#     if title is not None:
#         plt.title(title)
#     plt.imshow(array, interpolation='nearest', cmap=plt.cm.gray)
#     if show:
#         plt.show()

def display_network(A, cols=None, opt_normalize=True,
                    gray_color=True):
    if(gray_color):
        display_network_in_gray_mode(A,cols,opt_normalize)
    else:
        display_network_in_color_mode(A,cols,opt_normalize)

def display_network_in_gray_mode(A, cols=None, opt_normalize=False):
    n,m = A.shape
    sz  = int(np.sqrt(n))
    if(sz ** 2 != n):
        raise ValueError('Gray mode: When setting cols of an image'
                        ' to sqrt(len(A)), we found cols^2 ! len(A)')
    if(type(cols) == type(None)):
        cols = np.ceil(np.sqrt(m))
    if(opt_normalize):
        A = A - np.mean(A,axis=1,keepdims=True)
    cols = int(cols)
    rows = int(np.ceil(m / cols))
    figure, axes = plt.subplots(nrows = rows, ncols = cols)

    k = 0
    for axis in axes.flat:
        axis.set_frame_on(False)
        axis.set_axis_off()
        if(k >= m):
            continue
        axis.imshow(A[:,k].reshape(sz,sz),cmap = plt.cm.gray,
                             interpolation = 'nearest')
        k += 1

    plt.show()

def display_network_in_color_mode(A, cols=None, opt_normalize=True,file='visualize_W.png'):
    if(opt_normalize):
        A = A - np.mean(A,axis=1,keepdims=True)

    # n,m = A.shape
    # sz  = int(np.sqrt(n/3))
    #
    # if(sz ** 2 != int(n/3)):
    #     raise ValueError('Color mode: When setting cols of an image'
    #                     ' to sqrt(len(A)/3), we found cols^2 ! len(A)/3')
    # if(type(cols) == type(None)):
    #     cols = np.ceil(np.sqrt(m))
    # cols    = int(cols)
    # rows    = int(np.ceil(m / cols))
    # # todo



    cols = np.round(np.sqrt(A.shape[1]))

    channel_size = A.shape[0] / 3
    dim = np.sqrt(channel_size)
    dimp = dim + 1
    rows = np.ceil(A.shape[1] / cols)

    B = A[0:channel_size, :]
    C = A[channel_size:2 * channel_size, :]
    D = A[2 * channel_size:3 * channel_size, :]

    B = B / np.max(np.abs(B))
    C = C / np.max(np.abs(C))
    D = D / np.max(np.abs(D))

    # Initialization of the image
    image = np.ones(shape=(dim * rows + rows - 1, dim * cols + cols - 1, 3))

    for i in range(int(rows)):
        for j in range(int(cols)):
            # This sets the patch
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 0] = B[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 1] = C[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 2] = D[:, i * cols + j].reshape(dim, dim)

    image = (image + 1) / 2

    PIL.Image.fromarray(np.uint8(image * 255), 'RGB').save(file)


if __name__ == '__main__':
    import sample_images as sample
    patches = sample.sample_images('../data/IMAGES.mat','IMAGES',
                            patch_size = 8,num_patches = 10000)
    display_network(patches[:,0:145])
