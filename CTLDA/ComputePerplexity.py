# -*- coding: utf-8 -*-
'''
Created on 2014年5月14日

@author: ZhuJiahui506
'''
import os
import time
import numpy as np

def perplexity(read_directory1, read_directory2):
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    perp_list = []
    for i in range(file_number):
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        
        p_doc_word = np.dot(THETA, PHAI)
        
        this_prep = 0.0
        for j in range(len(p_doc_word)):
            log_v = np.log2([(x + 1) for x in p_doc_word[j]])
            #print np.sum(log_v)
            perp = np.power(2, (-1 * np.sum(log_v)))
            this_prep += perp
        
        this_prep = this_prep / len(p_doc_word)
        perp_list.append(this_prep)
        print this_prep
    print np.average(perp_list)
            

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset/CTLDA/doc_topic'
    read_directory2 = root_directory + u'dataset/CTLDA/topic_word'
    
    perplexity(read_directory1, read_directory2)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    