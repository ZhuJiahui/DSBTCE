# -*- coding: utf-8 -*-
'''
Created on 2014年4月10日

@author: ZhuJiahui506
'''

import os
import time
import jieba.posseg as jbp
from TextToolkit import quick_write_list_to_text
import jieba as jb

if __name__ == '__main__':
    start = time.clock()
    #now_directory = os.getcwd()
    #root_directory = now_directory + '/'
    
    ss = "高考作文题有错#你怎么看高考作文题出错？ 完全不应该，高考应是严谨神圣的  高考出题需要非常严谨！作文题目有问题，可能会影响考生答题！ 详情:http://t.cn/zHlLVBN"
    print len(ss)
    sss = jbp.cut(ss)
    a = []
    for each in sss:
        a.append(each.word + "/" + each.flag)
        print each
    print type(a[0])
    quick_write_list_to_text(a, '111.txt')
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    