# -*- coding: utf-8 -*-
'''
Created on 2014年5月14日

@author: ZhuJiahui506
'''

import numpy as np

def prf(cluster_tag, real_tag, reflect):
    cluster_number = len(reflect)
    p_list = []
    r_list = []
    f_list = []
    
    for i in range(cluster_number):
        
        correct = 0
        incorrect = 0
        origin_count = 0
        for j in range(len(cluster_tag)):
            if cluster_tag[j] == i:
                if reflect[cluster_tag[j]] == real_tag[j]:
                    correct += 1
                else:
                    incorrect += 1
            
            if real_tag[j] == reflect[i]:
                origin_count += 1
        
        this_p = np.true_divide(correct, (correct + incorrect))
        this_r = np.true_divide(correct, origin_count)
        this_f = 2 * this_p * this_r / (this_p + this_r)
        p_list.append(this_p)
        r_list.append(this_r)
        f_list.append(this_f)
    
    final_precision = np.average(p_list)
    final_recall = np.average(r_list)
    final_f = np.average(f_list)
    
    return final_precision, final_recall, final_f

if __name__ == '__main__':
    pass