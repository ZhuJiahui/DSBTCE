# -*- coding: utf-8 -*-
'''
Created on 2014年4月10日

@author: ZhuJiahui506
'''

import os
import time
import numpy as np
from TextToolkit import quick_write_list_to_text
from sklearn.cluster import KMeans

if __name__ == '__main__':
    start = time.clock()
    
    data = np.array([[1, 2.1, 0, 3],\
                     [1, 1.9, 0.1, 3],\
                     [4, 7, 3.1, 9.2],\
                     [3.8, 7, 3.0, 9.1],\
                     [0.9, 2.0, 0.01, 2.9]])
    
    
    
    w = np.zeros((len(data), len(data)))
    D = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            w[i, j] = 1.0 / (np.sum(np.abs(data[i] - data[j])) + 1.0)
        D[i, i] = 1.0 / np.sqrt(np.sum(w[i]))
        
    L = np.dot(np.dot(D, w), D)
    
    eigen_values, eigen_vectors = np.linalg.eig(L)
    print eigen_values
    print np.sum(eigen_values)
    
    # 按权值降序排序
    idx = eigen_values.argsort()
    idx = idx[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[idx] 
    
    new_data = eigen_vectors[0 : 2, :].transpose()
    print new_data
    
    k_means = KMeans(init='k-means++', n_clusters=2, n_init=20)
    k_means.fit(new_data)
    k_means_labels = k_means.labels_
    #k_means_cluster_centers = k_means.cluster_centers_
    #k_means_labels_unique = np.unique(k_means_labels)
    
    print k_means_labels


    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    