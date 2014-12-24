# -*- coding: utf-8 -*-
'''
Created on 2014年4月10日

@author: ZhuJiahui506
'''
import os
import time
from operator import itemgetter
from TextToolkit import quick_write_list_to_text, write_list_to_text_by_row


def generate_sort_index(read_directory, write_directory):
    '''
    排序索引的产生
    :param read_directory:
    :param write_directory:
    '''
    time_series = []
    item_index = []
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory)])
    
    for i in range(file_number):
        line_count = 1
        f = open(read_directory + '/' + str(i + 1) + '.txt', 'rb')
        line = f.readline()
        while line:
            #this_time = line.strip().split('\t')[2]
            this_time = time.mktime(time.strptime(line.strip().split('\t')[2], '%Y/%m/%d %H:%M'))
            time_series.append(this_time)
            item_index.append([str(i + 1), str(line_count)])  #索引下标均从1开始
            
            line = f.readline()
            line_count += 1
        f.close()
    
    #按时间升序排序
    tsi = zip(time_series, item_index)
    tsi1 = sorted(tsi, key = itemgetter(0))
            
    #选择对应的行号索引
    update_item_index = []
    for each in tsi1:
        update_item_index.append(each[1])
    
    write_list_to_text_by_row(update_item_index, write_directory + u'/update_item_index.txt')
    
    return update_item_index

def global_sort_by_time(update_item_index, read_directory, write_directory):
    
    print "Begin sorting." 
    print "May take a long time, Please Wait..."
    
    read_file_number = sum([len(files) for root, dirs, files in os.walk(read_directory)])
    
    segment = 50000

    total_length = len(update_item_index)
    segment_number = total_length / segment
    
    print "Total Segment %d ." % segment_number
    
    for i in range(segment_number):
        
        print "Segment %d ." % (i + 1)
        
        content_result = []
        for k in range(segment):
            content_result.append(" ")
        
        for j in range(read_file_number):
            f1 = open(read_directory + "/" +  str(j + 1) + ".txt", "rb")
            this_text_file = f1.readlines()
            f1.close()
            
            for l in range(segment):
                if update_item_index[segment * i + l][0] == str(j + 1):
                    content_result[l] = this_text_file[int(update_item_index[segment * i + l][1]) - 1].strip()
            
        quick_write_list_to_text(content_result, write_directory + "/" + str(i + 1) + ".txt")

    print "Global Sort Complete!!!"


if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory = root_directory + u'dataset/original_data'
    
    write_directory1 = root_directory + u'dataset/segment'
    write_directory2 = root_directory + u'dataset/segment/sort_original_data'
    
    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
        os.mkdir(write_directory2)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    
    update_item_index = generate_sort_index(read_directory, write_directory1)
        
    global_sort_by_time(update_item_index, read_directory, write_directory2)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    
