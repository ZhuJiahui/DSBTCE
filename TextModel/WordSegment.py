# -*- coding: utf-8 -*-
'''
Created on 2013年7月10日
Last on 2013年11月13日

@author: ZhuJiahui506
'''
import os
import jieba as jb
import jieba.posseg as jbp
from datetime import datetime

def get_stopwords1():
    '''
    获取停用词
    '''

    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    mark_stop = [x.strip().decode('gbk') for x in file (root_directory + "stopwords/mark_stop.txt")]
    english_stop = [x.strip().decode('gbk') for x in file (root_directory + "stopwords/Englishword.txt")]
    pre_CN_stop = [x.strip().decode('gbk') for x in file (root_directory + "stopwords/stop_word_pre.txt")]
    post_CN_stop = [x.strip().decode('gbk') for x in file(root_directory + "stopwords/stop_word.txt")]
    mul_stop = [x.strip().decode('gbk') for x in file(root_directory + "stopwords/stop_word_mul.txt") ]
    waste_content = [x.strip().decode('gbk') for x in file(root_directory + "stopwords/waste_content.txt") ]
    all_stop = [x.strip().decode('gbk') for x in file(root_directory + "stopwords/cn_stopwords.txt") ]
    return set(mark_stop + english_stop + pre_CN_stop + mul_stop + post_CN_stop + waste_content + all_stop + ["", " ", "  "])

def get_stopwords2():
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) +'/'
    english_stop = [x.strip().decode('gbk') for x in file (root_directory + "stopwords/Englishword.txt")]
    pre_CN_stop = [x.strip().decode('gbk') for x in file (root_directory + "stopwords/stop_word_pre.txt")]
    stopwords2 = set(english_stop + pre_CN_stop)
    return stopwords2
    
def word_segment(data, stopwords_list1):
    '''
    本处通用词只选stopwords_list1
    :param data:
    :param stopwords_list1:
    '''
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    jb.load_userdict(root_directory + "dataset/user_dict.txt")

    data = "".join(data)

    #segment = jb.cut(data)
    segment = jbp.cut(data)  #词性标注
    
    segment_list = []
    for item in segment:
        if (item.word not in stopwords_list1):
            segment_list.append(item.word.strip() + "/" + item.flag.strip())

    return segment_list;


if __name__ == '__main__':
    
    start = datetime.now()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    
    print '总共花了 %d 秒' % ((datetime.now() - start).seconds)
    print 'complete'
