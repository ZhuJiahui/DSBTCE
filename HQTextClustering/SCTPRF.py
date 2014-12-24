# -*- coding: utf-8 -*-
'''
Created on 2014年5月7日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text, get_text_to_single_list
import time
from HQTextClustering.PRF import prf


'''
Step 11
Text Clustering.
'''
def sct_prf(read_filename1, read_filename2, write_filename):
    
    cluster_tag = []
    real_tag = []
    
    get_text_to_single_list(cluster_tag, read_filename1)
    get_text_to_single_list(real_tag, read_filename2)

    cluster_tag = [int(x) for x in cluster_tag]
    real_tag = [int(x) for x in real_tag]
    
    reflect = [20, 20, 19]
    #reflect = [19, 20, 20]
    #reflect = [20, 19, 20]
    #reflect = [19, 20, 21, 22]
    #reflect = [20, 20, 22]

    p, r, f = prf(cluster_tag, real_tag, reflect)
    print p
    print r
    print f
    

    quick_write_list_to_text([str(p), str(r), str(f)], write_filename)


if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory = root_directory + u'dataset/hq_text_clustering/sct_measure'
    
    read_filename1 = root_directory + u'dataset/hq_text_clustering/sct_cluster_tag/46.txt'
    read_filename2 = root_directory + u'dataset/text_model/class_tag/46.txt'
    write_filename = write_directory + u'/prf46.txt'

    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)

    sct_prf(read_filename1, read_filename2, write_filename)
    
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
