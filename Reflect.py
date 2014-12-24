# -*- coding: utf-8 -*-
'''
Created on 2014年1月5日

@author: ZhuJiahui506
'''

def reflect_vsm_to_wordlist(vsm, word_list):
    '''
    Get the wordlist according to the vector space.
    :param vsm: 向量空间
    :param word_list: 单词列表
    '''
    
    result = []
    for i in range(len(vsm)):
        if float(vsm[i]) > 0.000001:
            result.append(word_list[i])
    
    return result

def reflect_vsm_to_wordlist2(vsm, word_list):
    '''
    Get the wordlist according to the vector space.
    Along with the term frequency,separated by '/'
    :param vsm:
    :param word_list:
    '''
    result = []
    for i in range(len(vsm)):
        if float(vsm[i]) > 0.05:
            result.append(word_list[i] + '/' + str(int(vsm[i])))
    
    return result

if __name__ == '__main__':
    pass