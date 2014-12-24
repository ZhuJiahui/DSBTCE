# -*- coding: utf-8 -*-
'''
Created on 2014年4月10日

@author: ZhuJiahui506
'''

import os
import time
from gensim import corpora, models, similarities
import logging
from TextToolkit import quick_write_list_to_text

if __name__ == '__main__':
    start = time.clock()
    #now_directory = os.getcwd()
    #root_directory = now_directory + '/'
    
    #ss = "高考作文题有错#你怎么看高考作文题出错？ 完全不应该，高考应是严谨神圣的  高考出题需要非常严谨！作文题目有问题，可能会影响考生答题！ 详情:http://t.cn/zHlLVBN"
    texts = [['total', 'death', 'toll', 'in', 'kunming', 'terror', 'attack', 'rises', 'to', '29'], \
             ['china', 'ready', 'for', 'annual', 'two', 'sessions'], \
             ['china', 'no', 'one', 'is', 'above', 'the', 'law', 'two', 'sessions'], \
             ['manhunt', 'after', 'terror', 'attack'], \
             ['kunming', 'restores', 'order', 'after', 'deadly', 'terror', 'attack']]
    texts2 = [['sessions'], ['china'], ['attack'], ['terror']]
    dictionary = corpora.Dictionary(texts)
    
    #print dictionary.token2id
    
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print corpus
    
    result = []
    for each in corpus:
        ss = [str(x) for x in each]
        result.append("+".join(ss))
    
    quick_write_list_to_text(result, 'corpus.txt')

    lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)
    print lda[corpus[1]]
    
    
    print lda.show_topics(topics=2)
    #print lda.show_topics(topics=2, formatted=False)[0]
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    