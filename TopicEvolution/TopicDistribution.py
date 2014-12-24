# -*- coding: utf-8 -*-
'''
Created on 2014年5月14日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text
import time
from operator import itemgetter

'''
Step 10
Central Topic LDA.
'''

def topic_distribution(read_directory1, read_directory2, write_directory1):
    
    gamma = 0.005
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(file_number):
        
        PHAI = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        
        if len(PHAI) >= 200:
            PHAI = np.array([PHAI])
            
        # 本片数据的词汇列表
        this_word_list = []
        f1 = open(read_directory2 + '/' + str(i + 1) + '.txt', 'rb')
        line = f1.readline()
        while line:
            this_word_list.append(line.split()[0])
            line = f1.readline()
        
        f1.close()
        
        real_topics = []
        for j in range(len(PHAI)):
            this_topic = []
            this_topic_weight = []
            for k in range(len(PHAI[j])):
                if PHAI[j][k] > gamma:
                    this_topic.append(this_word_list[k])
                    this_topic_weight.append(PHAI[j][k])
            
            tt = zip(this_topic, this_topic_weight)
            tt = sorted(tt, key = itemgetter(1), reverse=True)
            this_topic = []
            for each in tt:
                this_topic.append(str(each[1]) + '*' + str(each[0]))
            
            real_topics.append(" ".join(this_topic))

        quick_write_list_to_text(real_topics, write_directory1 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)
        

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory = root_directory + u'dataset/topic_evolution'
    
    read_directory1 = root_directory + u'dataset/CTLDA/stream_ct_word'
    read_directory2 = root_directory + u'dataset/text_model/select_words'
    write_directory1 = write_directory + u'/topic_distribution'

    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
        os.mkdir(write_directory1)

    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)


    topic_distribution(read_directory1, read_directory2, write_directory1)
      
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    