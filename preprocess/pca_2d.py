# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np

def pca(data,percent=0.99):
    U, S, _     = np.linalg.svd(data.dot(data.T) / data.shape[1])
    ind         = np.argsort(-S) # descent
    U           = U[:,ind]
    abs_S       = abs(S[ind])
    if(percent < 1): # in order to get a correct result under accuracy of float
        ind         = np.nonzero(np.cumsum(abs_S) / np.sum(abs_S) >= percent)[0]
        S           = S[0:ind[0]+1]
        U           = U[:,0:ind[0]+1]
    return U,S

def whitening(U,S,data_rot,eps=1e-5,method='PCAWhitening'):
    data_whitening  = np.diag(1 / np.sqrt(abs(S) + eps)).T.dot((data_rot))
    if(method.lower() == 'ZCAWhitening'.lower()):
        data_whitening = U.dot(data_whitening)
    return data_whitening

def load_data(file_name,data_split=None):
    file = open(file_name)
    data = []
    for line in file.readlines():
        line = line.strip()
        if(line == ''):
            continue
        data.append([float(num) for num in line.split(data_split)])
    return np.array(data)


def plot_data(data,PCs=[]):
    import matplotlib.pyplot as plt
    if(len(PCs) < 2):
        PCs = np.atleast_2d(PCs).T
    if(data.shape[0] > 1):
        plt.plot(data[0,],data[1,],'o')
    else:
        plt.plot(data[0,],data[0,],'o')
    if(PCs.size > 0):
        for i in range(PCs.shape[1]):
            plt.plot([0,PCs[0,i]],[0,PCs[1,i]],'r-')
    plt.show()

if __name__ == "__main__":
    data            = load_data('../data/pcaData.txt',data_split=None)
    PCs, eigval     = pca(data,percent=0.99)
    data_rot        = PCs.T.dot(data)
    data_hat        = PCs.dot(data_rot)
    data_whitening  = whitening(PCs,eigval,data_rot,method='PCAWhitening')
    print(data_whitening.dot(data_whitening.T)/data_whitening.shape[1])
    plot_data(data_whitening)