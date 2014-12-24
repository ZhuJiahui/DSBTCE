# -*- coding: utf-8 -*-
'''
Created on 2014年5月6日

@author: ZhuJiahui506
'''

import os
import time
from operator import itemgetter
from TextToolkit import quick_write_list_to_text, get_text_to_complex_list,\
    get_text_to_single_list

'''
Step 5
Count TF
'''

def batch_count_tf(read_directory1, read_directory2, write_directory):
    '''
    
    :param read_directory1:
    :param read_directory2:
    :param write_directory:
    '''

    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(file_number):
        each_weibo_fenci = [] 
        all_weibo_fenci = []
        
        get_text_to_complex_list(each_weibo_fenci, read_directory1 + '/' + str(i + 1) + '.txt', 0)
        get_text_to_single_list(all_weibo_fenci, read_directory2 + '/' + str(i + 1) + '.txt')
        
        tf_dict = {}  #词频TF字典
        for key in all_weibo_fenci:
            tf_dict[key] = 0
            
        for row in range(len(each_weibo_fenci)):
            for j in range(len(each_weibo_fenci[row])):
                try:
                    tf_dict[each_weibo_fenci[row][j]] += 1
                except KeyError:
                    tf_dict[each_weibo_fenci[row][j]] = 0
        
        #词频列表
        value_list = []
        for key in all_weibo_fenci:
            value_list.append(tf_dict[key])
        
        # 按词频降序排序
        va = zip(all_weibo_fenci, value_list)
        va = sorted(va, key = itemgetter(1), reverse = True)    
        
        result_all = []
        for each in va:
            result_all.append(each[0] + " " + str(each[1]))
        
        quick_write_list_to_text(result_all, write_directory + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset/text_model/content1'
    read_directory2 = root_directory + u'dataset/text_model/all_weibo_word'
    write_directory = root_directory + u'dataset/text_model/tf_all'


    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
    
    batch_count_tf(read_directory1, read_directory2, write_directory)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    