# -*- coding: utf-8 -*-
'''
Created on 2014年5月7日

@author: ZhuJiahui506
'''

import os
import time
import numpy as np
from sklearn.cluster import KMeans
from KLD import SKLD

def get_max_eig(L, m, withev=False):
    '''
    
    :param L:
    :param m:
    '''
    eigen_values, eigen_vectors = np.linalg.eig(L)
    
    # 降序排序
    idx = eigen_values.argsort()
    idx = idx[::-1]
    
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[idx] 
    
    max_eigen_values = eigen_values[0 : m]
    max_eigen_vectors = eigen_vectors[0 : m, :]
    
    #正向化
    max_eigen_vectors = np.abs(max_eigen_vectors)
    
    if withev:
        return max_eigen_values, max_eigen_vectors
    else:
        return max_eigen_vectors
    
def sym(X):
    return (X + X.transpose()) / 2.0

def RMSE(X1, X2):
    eX = X1 - X2
    error = 0.0
    for i in range(len(eX)):
        for j in range(len(eX[0])):
            error += (eX[i, j] * eX[i, j])
    
    return np.sqrt(error / len(eX) / len(eX[0]))

def spectral_cluster(data, cluster_number):
    dimension = len(data)
    withev = False
    
    W1 = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            W1[i, j] = 1.0 / (SKLD(data[i], data[j]) + 1.0)
            W1[j, i] = W1[i, j]
    
    D1 = np.zeros((dimension, dimension))
   
    for i in range(dimension):
        D1[i, i] = 1.0 / np.sqrt(np.sum(W1[i]))
  
    L1 = np.dot(np.dot(D1, W1), D1)
 
    U1 = get_max_eig(L1, cluster_number, withev)
    
    U1 = U1.transpose()
    k_means = KMeans(init='k-means++', n_clusters=cluster_number, n_init=10)
    k_means.fit(U1)
    k_means_labels = k_means.labels_
    
    return k_means_labels

def spectral_cluster2(W1, cluster_number):
    dimension = len(W1)
    withev = False
    
    D1 = np.zeros((dimension, dimension))
   
    for i in range(dimension):
        D1[i, i] = 1.0 / np.sqrt(np.sum(W1[i]))
  
    L1 = np.dot(np.dot(D1, W1), D1)
 
    U1 = get_max_eig(L1, cluster_number, withev)
    
    U1 = U1.transpose()
    k_means = KMeans(init='k-means++', n_clusters=cluster_number, n_init=10)
    k_means.fit(U1)
    k_means_labels = k_means.labels_
    
    return k_means_labels
    

def co_training_spectral_cluster(W1, W2, cluster_number, iter):
    '''
    
    :param W1:
    :param W2:
    :param cluster_number:
    :param iter:
    '''
    
    dimension = len(W1)
    
    D1 = np.zeros((dimension, dimension))
    D2 = np.zeros((dimension, dimension))
    for i in range(dimension):
        D1[i, i] = 1.0 / np.sqrt(np.sum(W1[i]))
        D2[i, i] = 1.0 / np.sqrt(np.sum(W2[i]))
        
    L1 = np.dot(np.dot(D1, W1), D1)
    L2 = np.dot(np.dot(D2, W2), D2)
    
    U1 = get_max_eig(L1, cluster_number, withev=False)
    U2 = get_max_eig(L2, cluster_number, withev=False)
    
    for mmi in range(iter):
        FW1 = sym(np.dot(np.dot(U2.transpose(), U2), W1))
        FW2 = sym(np.dot(np.dot(U2.transpose(), U2), W2))
        '''
        rmse = RMSE(FW1, W1)
        print "RMSE", rmse
        if rmse < 0.01:
            break
        '''
        for j in range(dimension):
            D1[j, j] = 1.0 / np.sqrt(np.sum(FW1[j]))
            D2[j, j] = 1.0 / np.sqrt(np.sum(FW2[j]))
        
        L1 = np.dot(np.dot(D1, FW1), D1)
        L2 = np.dot(np.dot(D2, FW2), D2)
        
        U1 = get_max_eig(L1, cluster_number, withev=False)
        U2 = get_max_eig(L2, cluster_number, withev=False)
        
        W1 = FW1
        W2 = FW2
    
    U1 = U1.transpose()
    k_means = KMeans(init='k-means++', n_clusters=cluster_number, n_init=10)
    k_means.fit(U1)
    k_means_labels = k_means.labels_
    
    return k_means_labels

if __name__ == '__main__':
    W1 = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
    W2 = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.3], [0.1, 0.3, 1.0]])
    
    cluster_tag = co_training_spectral_cluster(W1, W2, 2, iter=3)