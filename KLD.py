# -*- coding: utf-8 -*-
'''
Created on 2014年5月7日

@author: ZhuJiahui506
'''

import numpy as np

def SKLD(x1, x2):
    '''
    对称KLD
    :param x1:
    :param x2:
    '''
    x1 = x1 + 0.001
    x2 = x2 + 0.001
    x1 = x1 / np.sum(x1)
    x2 = x2 / np.sum(x2)
    
    KLD1 = 0.0
    KLD2 = 0.0
    for i in range(len(x1)):
        KLD1 += x1[i] * np.log(x1[i] / x2[i])
        KLD2 += x2[i] * np.log(x2[i] / x1[i])

    return ((KLD1 + KLD2) / 2.0)
    

if __name__ == '__main__':
    pass