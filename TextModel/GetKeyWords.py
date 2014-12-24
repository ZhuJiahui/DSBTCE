# -*- coding: utf-8 -*-
'''
Created on 2014年5月6日

@author: ZhuJiahui506
'''
import os
from TextToolkit import get_text_to_complex_list, quick_write_list_to_text
import jieba.analyse
import time


'''
Step 6
Get Key Words
'''

def get_key_words(read_directory, write_directory1, write_directory2):
    '''
    
    :param read_directory:
    :param write_directory1:
    :param write_directory2:
    '''

    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory)])
    
    for i in range(file_number):
        each_weibo_fenci = []        
        get_text_to_complex_list(each_weibo_fenci, read_directory + '/' + str(i + 1) + '.txt', 0)
        
        key_words = []
        all_key_words =  []
        for row in range(len(each_weibo_fenci)):
            word_entity = []

            for each in each_weibo_fenci[row]:
                word_entity.append(each.split('/')[0])

            tags = jieba.analyse.extract_tags(" ".join(word_entity), 3)
            key_words.append(" ".join(tags))
            
            for word in " ".join(tags).split():
                if word not in all_key_words:
                    all_key_words.append(word)
        
        quick_write_list_to_text(key_words, write_directory1 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(all_key_words, write_directory2 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory = root_directory + u'dataset/text_model/content1'
    write_directory1 = root_directory + u'dataset/text_model/key_words'
    write_directory2 = root_directory + u'dataset/text_model/all_key_words'


    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    
    get_key_words(read_directory, write_directory1, write_directory2)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
