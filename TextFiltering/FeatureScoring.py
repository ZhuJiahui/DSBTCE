# -*- coding: utf-8 -*-
'''
Created on 2014年4月11日

@author: ZhuJiahui506
'''
import os
import time
import re
import numpy as np
from TextToolkit import quick_write_list_to_text

'''
Warning The original data was not oerderd by EM weight.
Step 1
'''

def score_normalize(data_list, per, enlar):

    max_data = np.max(data_list)
    min_data = np.min(data_list)
    percent = (max_data - min_data) * per
    
    result = []

    for i in range(len(data_list)):
        result.append((data_list[i] - min_data + percent) / (max_data - min_data + percent) * enlar)
    
    return result

def square_score_normalize(data_list, per, enlar):
    data_list = np.power(data_list, 2)
    result = score_normalize(data_list, per, enlar)
    
    return result

def exp_score_normalize(data_list, per, enlar):
    data_list = np.exp2(np.array(data_list) / 20.0)
    result = score_normalize(data_list, per, enlar)
    
    return result

def log_score_normalize(data_list, per, enlar):

    log_data = []
    for each in data_list:
        if each <= 0:
            log_data.append(0.0)
        else:
            log_data.append(np.log(each + 1.0))
    
    result = score_normalize(log_data, per, enlar)
    
    return result

def t_log_score_normalize(data_list, per, enlar):
    '''
    log10 normalize
    :param data_list:
    :param per:
    :param enlar:
    '''

    log_data = []
    for each in data_list:
        if each <= 0:
            log_data.append(0.0)
        else:
            log_data.append(np.log10(each + 1.0))
    
    result = score_normalize(log_data, per, enlar)
    
    return result

def feature_scoring(read_directory, write_directory):
     
    pa = 'http://[\w\.\*\?-_#/]*(?=\b)?'
    per = 0.05
    enlar = 100.0

    print 'Text Scoring for Filtering...'
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory)])
    
    for i in range(file_number):
        
        weibo_forwards = []
        weibo_comments = []
        weibo_length = []
        weibo_urls = []
        
        url_dict = {}
        
        #weibo_content = []
        f = open(read_directory + '/' + str(i + 1) + '.txt', 'rb')
        line = f.readline()
        while line:
            this_line = line.strip().split('\t')
            
            weibo_forwards.append(int(this_line[3]))
            weibo_comments.append(int(this_line[4]))
            weibo_length.append(len(this_line[6]))
            
            try:             
                matches = re.findall(pa, this_line[6])
                
                for ma in matches:
                    if ma in url_dict:
                        url_dict[ma] += 1
                    else:
                        url_dict[ma] = 1
                weibo_urls.append(matches)
            except Exception, e:
                print e
                print "Exception at segment %d" % (i + 1)
                weibo_urls.append([])
            
            line = f.readline()
        f.close()
        
        url_score = []
        for j in range(len(weibo_forwards)):
            if len(weibo_urls[j]) > 0:
                sum_score = 0
                for each in weibo_urls[j]:
                    sum_score += url_dict[each]
                url_score.append(sum_score)
            else:
                url_score.append(0)
        
        # 特征打分值归一化
        forwards_norm = log_score_normalize(weibo_forwards, per, enlar)
        comments_norm = log_score_normalize(weibo_comments, per, enlar)
        length_norm = square_score_normalize(weibo_length, per, enlar)
        urls_norm = t_log_score_normalize(url_score, per, enlar)

        
        result_all = []
        for j in range(len(weibo_forwards)):
            result_all.append(str(forwards_norm[j]) + " " + str(comments_norm[j]) + " " + str(length_norm[j]) + " " + str(urls_norm[j]))
   
        quick_write_list_to_text(result_all, write_directory + '/' + str(i + 1) + '.txt')

  
if __name__=='__main__':
    
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory = root_directory + u'dataset/segment/sort_original_data'
    write_directory1 = root_directory + u'dataset/text_filtering'
    write_directory2 = root_directory + u'dataset/text_filtering/feature_score'
    
    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
        os.mkdir(write_directory2)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    
    feature_scoring(read_directory, write_directory2)

    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    