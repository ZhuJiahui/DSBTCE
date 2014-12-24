# -*- coding: utf-8 -*-
'''
Created on 2014年4月25日

@author: ZhuJiahui506
'''
import os
import time
from operator import itemgetter
from TextToolkit import quick_write_list_to_text, get_text_to_single_list

'''
Step 3
Compute EM weight of each Weibo and ordered by its EM weights to generate the high quality data. 
So the original data was changed.
'''


def generate_high_quality_data(read_directory1, read_directory2, read_directory3, write_directory):
    '''
    Linear fusion
    :param read_directory1:
    :param read_directory2:
    :param read_directory3:
    :param write_directory:
    '''
    K = 3000

    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    for i in range(file_number):
        em_weights = []
    
        coefficients_string = [] 
        get_text_to_single_list(coefficients_string, read_directory1 + '/' + str(i + 1) + '.txt')   
        coefficients = [float(x) for x in coefficients_string]
        
        f = open(read_directory2 + '/' + str(i + 1) + '.txt', 'r')
        line = f.readline()
        while line:
            each_line = line.split()
            this_em = 0.0
            for j in range(len(coefficients)):
                this_em += float(each_line[j]) * coefficients[j]
            
            em_weights.append(this_em)
            
            line = f.readline()
        f.close()
        
        this_weibo = []
        time_series = []
        this_text = []
        #get_text_to_single_list(this_weibo, read_directory3 + '/' + str(i + 1) + '.txt')
        
        f = open(read_directory3 + '/' + str(i + 1) + '.txt', 'rb')
        line = f.readline()
        while line:
            this_time = time.mktime(time.strptime(line.strip().split('\t')[2], '%Y/%m/%d %H:%M'))
            time_series.append(this_time)
            this_weibo.append(line.strip())
            try:
                this_text.append(line.strip().split('\t')[6])
            except:
                this_text.append(" ")
            
            line = f.readline()
        f.close()
        
        # 按EM值排序
        ttte = zip(this_weibo, time_series, this_text, em_weights)
        ttte1 = sorted(ttte, key = itemgetter(3), reverse = True)
        
        this_weibo = []
        time_series = []
        this_text = []
        em_weights = []
        
        line_count = 0
        for each in ttte1:
            if each[2] not in this_text and len(each[2]) >= 150:
                this_weibo.append(each[0]+'\t'+str(each[3]))
                time_series.append(each[1])
                this_text.append(each[2])
                line_count += 1
                
                if line_count >= K:
                    break
        
        # 再按时间升序排序
        twts = zip(this_weibo, time_series)
        twts1 = sorted(twts, key = itemgetter(1))
        
        this_weibo = []
        time_series = []
        this_text = []
        
        for each in twts1:
            this_weibo.append(each[0])

        quick_write_list_to_text(this_weibo, write_directory + '/' + str(i + 1) + '.txt')
    
if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset/text_filtering/em_coefficients'
    read_directory2 = root_directory + u'dataset/text_filtering/feature_score'
    read_directory3 = root_directory + u'dataset/segment/sort_original_data'
    write_directory = root_directory + u'dataset/text_filtering/high_quality_data'

    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
    
    generate_high_quality_data(read_directory1, read_directory2, read_directory3, write_directory)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    