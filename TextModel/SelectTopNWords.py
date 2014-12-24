# -*- coding: utf-8 -*-
'''
Created on 2014年5月6日

@author: ZhuJiahui506
'''

import os
from operator import itemgetter
import numpy as np
from TextToolkit import get_text_to_complex_list, get_text_to_single_list, quick_write_list_to_text
import time

'''
Step 7
Get the top N words for the high quality data(8000 Weibo). According to the TF, user_dict, keywords and word tag.
'''

def select_top_N_words(read_directory1, read_directory2, read_filename, write_directory):
    '''
    选取前N个词作为高质量的特征词汇
    :param read_directory1:
    :param read_directory2:
    :param read_filename:
    :param write_directory:
    '''
    N = 500
    
    #根据词性分配权值
    score_dict = {"nr":1.0, "nr1":0.5, "nr2":0.75, "nrt":1.0, "nrf":1.0, "ns":1.0, "nsf":1.0, "nt":1.0, \
                   "nz":1.0, "nl":0.5, "ng":0.5, "n":0.9, "t":0.5, "tg":0.5, "s":0.3, "f":0.3, "j":0.5, \
                   "v":0.7, "vd":0.6, "vn":0.9, "vshi":0.0, "vyou":0.0, "vf":0.3, "vx":0.3, "vi":0.7, \
                   "vl":0.3, "vg":0.5, "a":0.6, "ad":0.3, "an":0.9, "ag":0.5, "al":0.3, "b":0.3, "bl":0.2, \
                    "z":0.9, "zg":0.3, "r":0.3, "rr":0.3, "rz":0.3, "rzt":0.3, "rzs":0.3, "rzv":0.3, "ry":0.2, \
                    "ryt":0.2, "rys":0.2, "ryv":0.2, "rg":0.2, "m":0.2, "mq":0.5, "q":0.6, "qv":0.7, "qt":0.7, \
                    "d":0.4, "p":0.0, "pba":0.0, "pbei":0.0, "c":0.0, "cc":0.0, "u":0.0, "ug":0.0, "e":0.0, \
                    "y":0.0, "o":0.0, "h":0.0, "k":0.0, "x":0.1, "xx":0.0, "xu":0.9, "w":0.0, "l":0.6, "i":0.6, \
                    "g":0.0, "vq":0.0, "nrfg":0.75, "dg":0.0, "mg":0.2, "yg":0.0, "eng":0.1}
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(file_number):
        each_word_tf = [] 
        key_words = []
        
        select_word = []
        word_score = []
    
        user_dict = []
        
        get_text_to_complex_list(each_word_tf, read_directory1 + '/' + str(i + 1) + '.txt', 0)
        
        get_text_to_single_list(key_words, read_directory2 + '/' + str(i + 1) + '.txt')
    
        f = open(read_filename, 'r')
        line = f.readline()
        while line:
            user_dict.append(line.split()[0])
            line = f.readline()
        f.close()
        
        for j in range(len(each_word_tf)):
            word_entity = each_word_tf[j][0].split('/')[0]
            word_tag = each_word_tf[j][0].split('/')[1]
            if word_entity in user_dict:
                #用户词典中的词分配高权值
                select_word.append(word_entity)
                word_score.append(np.log(float(each_word_tf[j][1])) * 1.0 * 1.0)
            elif word_entity in key_words and word_tag != 'eng':
                #关键词也分配高权值
                select_word.append(word_entity)
                try:
                    word_score.append(np.log(float(each_word_tf[j][1])) * score_dict[word_tag] * 1.0)
                except KeyError:
                    word_score.append(float(0.0))
                    
            else:
                #其余词汇乘以0.6
                select_word.append(word_entity)
                try:
                    word_score.append(np.log(float(each_word_tf[j][1])) * score_dict[word_tag] * 0.50)
                except KeyError:
                    word_score.append(float(0.0))
        
        # 按权值降序排序
        sw = zip(select_word, word_score)
        sw = sorted(sw, key = itemgetter(1), reverse = True)    
        
        result_all = []
        count_number = 1
        for each in sw:
            result_all.append(each[0] + " " + str(each[1]))
            count_number += 1
            if count_number > N:
                break
        
        quick_write_list_to_text(result_all, write_directory + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset/text_model/tf_all'
    read_directory2 = root_directory + u'dataset/text_model/all_key_words'
    read_filename = root_directory + u'dataset/user_dict.txt'
    write_directory = root_directory + u'dataset/text_model/select_words'


    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)

    
    select_top_N_words(read_directory1, read_directory2, read_filename, write_directory)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
