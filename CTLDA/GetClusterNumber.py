# -*- coding: utf-8 -*-
'''
Created on 2014年5月7日

@author: ZhuJiahui506
'''

import os
import numpy as np
import time

def get_cluster_number(X):
    '''
    估计聚类个数
    :param X: 数据之间的相似度矩阵 维度小于1000
    '''
    
    D = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        D[i, i] = 1.0 / np.sqrt(np.sum(X[i]))
    
    L = np.dot(np.dot(D, X), D)
    
    eigen_values, eigen_vectors = np.linalg.eig(L)
    
    # 按特征值降序排序
    idx = eigen_values.argsort()
    idx = idx[::-1]
    eigen_values = eigen_values[idx]
    
    content = np.sum(eigen_values) * 0.45
    
    now_sum = 0.0
    
    cluster_number = 0
    for i in range(len(eigen_values)):
        now_sum += eigen_values[i]
        cluster_number += 1
        if now_sum > content:
            break
    
    return cluster_number
    
if __name__ == '__main__':
    start = time.clock()
    data = np.array([[1, 2.1, 0, 3],\
                     [1, 1.9, 0.1, 3],\
                     [4, 7, 3.1, 9.2],\
                     [3.8, 7, 3.0, 9.1],\
                     [0.9, 2.0, 0.01, 2.9]])
    
    
    
    w = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            w[i, j] = 1.0 / (np.sum(np.abs(data[i] - data[j])) + 1.0)
    
    c = get_cluster_number(w)
    print c
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    