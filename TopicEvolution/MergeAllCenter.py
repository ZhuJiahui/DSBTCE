# -*- coding: utf-8 -*-
'''
Created on 2014年3月28日

@author: ZhuJiahui506
'''

import os
import time
from TextToolkit import quick_write_list_to_text
import numpy as np
from Reflect import reflect_vsm_to_wordlist

def merge_all_center(read_directory1, read_directory2, start_batch, end_batch):
    #文件总数
    file_number = np.sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    new_word_list = []
    
    for i in range(start_batch, end_batch):
        word_list = []
        f1 = open(read_directory2 + '/' + str(i) + '.txt', 'rb')
        line = f1.readline()
        while line:
            word_list.append(line.split()[0])
            line = f1.readline()
        
        f1.close()
        
        center = np.loadtxt(read_directory1 + '/' + str(i) + '.txt')
        if len(center) >= 200:
            center = np.array([center])
        
        for each in center:
            word_result = reflect_vsm_to_wordlist(each, word_list)
            for word in set(word_result).difference(new_word_list):
                new_word_list.append(word)
    
    result = []
    all_count = 0
    for i in range(start_batch, end_batch):
        word_list = []
        f1 = open(read_directory2 + '/' + str(i) + '.txt', 'rb')
        line = f1.readline()
        while line:
            word_list.append(line.split()[0])
            line = f1.readline()
        
        f1.close()
        
        center = np.loadtxt(read_directory1 + '/' + str(i) + '.txt')
        if len(center) >= 200:
            center = np.array([center])
        
        this_result = []
        for each in center:
            tf_dict = {}
            for k in range(len(each)):
                if each[k] > 0.000001:
                    tf_dict[word_list[k]] = each[k]
                
            tf_dict2 = {}
            for each1 in new_word_list:
                if each1 in tf_dict.keys():
                    tf_dict2[each1] = tf_dict[each1]
                else:
                    tf_dict2[each1] = 0
            
            this_line = np.zeros(len(new_word_list))
            count = 0
            for key in new_word_list:
                this_line[count] = tf_dict2[key]
                count += 1
            
            #每一行合并为字符串，方便写入
            this_result.append(this_line)
            all_count += 1
        
        result.append(this_result)
        #quick_write_list_to_text(result, write_directory + '/' + str(i + 1) + '.txt')
    
    return result, new_word_list, all_count
    #quick_write_list_to_text(new_word_list, write_filename)
        

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset/global_em/cluster_center1'
    read_directory2 = root_directory + u'dataset/segment/top_n_words'
    #read_directory1 = root_directory + u'dataset/global_kmeans/cluster_center1'
    #read_directory2 = root_directory + u'dataset/segment/top_n_words'

    write_directory = root_directory + u'dataset/global_em/merge_cluster_center'
    write_filename = root_directory + u'dataset/global_em/new_word_list.txt'
    #write_directory = root_directory + u'dataset/global_kmeans/merge_cluster_center'
    #write_filename = root_directory + u'dataset/global_kmeans/new_word_list.txt'
    
    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
    
    #merge_all_center(read_directory1, read_directory2, write_directory, write_filename)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'