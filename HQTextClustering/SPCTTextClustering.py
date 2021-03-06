# -*- coding: utf-8 -*-
'''
Created on 2014年5月7日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text
import time
from operator import itemgetter


'''
Step 11
Text Clustering.
'''
def hq_text_clustering(read_directory1, read_directory2, read_directory3, write_directory1, write_directory2):
    
    gamma = 0.01
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(file_number):
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        
        # 本片数据的词汇列表
        this_word_list = []
        f1 = open(read_directory3 + '/' + str(i + 1) + '.txt', 'rb')
        line = f1.readline()
        while line:
            this_word_list.append(line.split()[0])
            line = f1.readline()
        
        f1.close()
        
        if len(PHAI) >= 200:
            PHAI = np.array([PHAI])
        
        cluster_tag = []
        
        for j in range(len(THETA)):
            cluster_tag.append(str(np.argmax(THETA[j])))
        
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
                this_topic.append(each[0])
            
            real_topics.append(" ".join(this_topic))

        quick_write_list_to_text(cluster_tag, write_directory1 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(real_topics, write_directory2 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)


if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory = root_directory + u'dataset/hq_text_clustering'
    
    read_directory1 = root_directory + u'dataset/CTLDA/sp_doc_ct'
    read_directory2 = root_directory + u'dataset/CTLDA/sp_ct_word'
    read_directory3 = root_directory + u'dataset/text_model/select_words'
    write_directory1 = write_directory + u'/sp_cluster_tag'
    write_directory2 = write_directory + u'/sp_real_topic'

    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
        os.mkdir(write_directory1)
        os.mkdir(write_directory2)

    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)

    hq_text_clustering(read_directory1, read_directory2, read_directory3, write_directory1, write_directory2)
    
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
