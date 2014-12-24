# -*- coding: utf-8 -*-
'''
Created on 2014年5月7日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text, write_matrix_to_text
import time
from KLD import SKLD
from CTLDA.GetClusterNumber import get_cluster_number
from CTLDA.CoTrainingSP import co_training_spectral_cluster, spectral_cluster

'''
Step 10
Central Topic LDA.
'''

def merge_space(word_list1, word_list2, p1, p2):
    new_word_list = word_list2
    for word in set(word_list1).difference(new_word_list):
        new_word_list.append(word)
    
    new_p2 = np.zeros((len(p2), len(new_word_list)))
    new_p2[:, 0 : len(p2[0])] = p2
    
    new_p1 = np.zeros((len(p1), len(new_word_list)))

    for j in range(len(p1)):
        p1_dict = {}
        for k in range(len(p1[j])):
            if p1[j][k] > 0.0001:
                p1_dict[word_list1[k]] = p1[j][k]
                
        p1_dict2 = {}
        for each1 in new_word_list:
            if each1 in p1_dict.keys():
                p1_dict2[each1] = p1_dict[each1]
            else:
                p1_dict2[each1] = 0
        
        for k in range(len(new_word_list)):
            new_p1[j, k] = p1_dict2[new_word_list[k]]
            
    return new_p1, new_p2, new_word_list
            

def stream_CT_LDA(read_directory1, read_directory2, read_directory3, write_directory1, write_directory2, write_directory3, write_filename):
    
    gamma = 0.1
    s_lambda = 0.7
    
    # 时间窗口
    q = 4
    
    ct_window = []
    ct_num_window = []
    ct_wordlist_window = []
    
    run_time = []
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(file_number):
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        
        # 视图1，根据词汇分布计算潜在主题之间的相似度
        W1 = np.zeros((len(PHAI), len(PHAI)))
        for j in range(len(PHAI)):
            for k in range(j, len(PHAI)):
                W1[j, k] = 1.0 / (SKLD(PHAI[j], PHAI[k]) + 1.0)
                W1[k, j] = W1[j, k]
        
        # 估计聚类数目
        cluster_number = get_cluster_number(W1)
        print cluster_number
        
        # 本片数据的词汇列表
        this_word_list = []
        f1 = open(read_directory3 + '/' + str(i + 1) + '.txt', 'rb')
        line = f1.readline()
        while line:
            this_word_list.append(line.split()[0])
            line = f1.readline()
        
        f1.close()
        
        start = time.clock()
        if i < q or np.mod(i, q) == 0: 
        
            # 视图2，根据相关微博文本集合计算潜在主题之间的相似度
            W2 = np.zeros((len(PHAI), len(PHAI)))
        
            related_weibo_list = []
            for j in range(len(PHAI)):
                related_weibo_list.append([])
            
            for j in range(len(THETA)):
                for k in range(len(THETA[0])):
                    if THETA[j, k] >= gamma:
                        related_weibo_list[k].append(j)
        
            for j in range(len(PHAI)):
                for k in range(j, len(PHAI)):
                    numerator = len(set(related_weibo_list[j]) & set(related_weibo_list[k]))
                    denominator = len(set(related_weibo_list[j]) | set(related_weibo_list[k]))
                    if j == k:
                        W2[j, k] = 1.0
                        W2[k, j] = 1.0
                    elif denominator == 0.0:
                        W2[j, k] = 0.0
                        W2[k, j] = 0.0
                    else:
                        W2[j, k] = np.true_divide(numerator, denominator)
                        W2[k, j] = W2[j, k]

            max_iter = 20
            
            cluster_tag = co_training_spectral_cluster(W1, W2, cluster_number, max_iter)
            
            # 聚类分析
            center_topic = np.zeros((cluster_number, len(PHAI[0])))
            each_cluster_number = np.zeros(cluster_number, int)
        
            weibo_topic_similarity = np.zeros((cluster_number, len(THETA)))
            THETA = THETA.transpose()
            
            for j in range(len(cluster_tag)):
                center_topic[cluster_tag[j]] += PHAI[j]
                each_cluster_number[cluster_tag[j]] += 1
            
                weibo_topic_similarity[cluster_tag[j]] += THETA[j]
            
            #
            for j in range(cluster_number):
                center_topic[j] = center_topic[j] / each_cluster_number[j]
        
            weibo_topic_similarity = weibo_topic_similarity.transpose()
        
        else:
            # 回溯一个数据片
            temp_ct = np.zeros((cluster_number, len(PHAI[0])))
            
            if len(ct_window[-1]) >= cluster_number:
                idx = ct_num_window[-1].argsort()
                idx = idx[::-1]
                
                temp_ct = ct_window[-1][idx][0 : cluster_number, :]
            else:
                temp_ct[0 : len(ct_window[-1]), :] = ct_window[-1]
                
            # 合并向量空间
            new_temp_ct, new_this_lt, new_word_list = merge_space(ct_wordlist_window[-1], this_word_list, temp_ct, PHAI)
            
            
            
            #计算当前潜在主题与前一片的中心主题之间的相似度
            lt_ct_similarity = np.zeros((len(new_this_lt), len(new_temp_ct)));
            for j in range(len(new_this_lt)):
                for k in range(len(new_temp_ct)):
                    lt_ct_similarity[j, k] = 1.0 / (SKLD(new_this_lt[j], new_temp_ct[k]) + 1.0)
            
            #print lt_ct_similarity
            
            cluster_tag = []
            
            new_part_lt = []  #原空间(500维)下的本数据片的新出现的潜在主题
            last_part_lt = []  #原空间下的与上一数据片中的中心主题比较相似的潜在主题,二维
            for j in range(len(new_temp_ct)):
                last_part_lt.append([])
                
            for j in range(len(new_this_lt)):
                if np.max(lt_ct_similarity[j]) < s_lambda:
                    new_part_lt.append(PHAI[j])
                    cluster_tag.append(-1)  #新类编号
                else:
                    max_index = np.argmax(lt_ct_similarity[j])
                    last_part_lt[max_index].append(PHAI[j])
                    cluster_tag.append(max_index)
            
            empty_count = 0
            this_last_ct = []
            this_last_ct_count = []
            #for j in range(len(new_this_lt)):
            for j in range(len(last_part_lt)):
                
                if len(last_part_lt[j]) == 0:
                    empty_count += 1
                else:
                    temp_this_ct = np.zeros(len(PHAI[0]))
                    temp_this_ct_count = 0
                    for k in range(len(last_part_lt[j])):
                        temp_this_ct += last_part_lt[j][k]
                        temp_this_ct_count += 1
                    
                    this_last_ct.append(temp_this_ct / temp_this_ct_count)
                    this_last_ct_count.append(temp_this_ct_count)
            
            
            center_topic = np.zeros((cluster_number, len(PHAI[0])))
            each_cluster_number = np.zeros(cluster_number, int)
            
            print "empty_number" , empty_count
            
            '''
            分情况讨论中心主题更新
            '''
            #new_part_it_number = cluster_number - (len(new_this_lt) - empty_count)
            new_part_it_number = cluster_number - (len(last_part_lt) - empty_count)
            print "new_part_it_number" , new_part_it_number
            if new_part_it_number == 0 and len(new_part_lt) == 0:
                #直接将上一片的主题作为本片的主题
                #每片求均值
                for j in range(len(this_last_ct)):
                    center_topic[j] = this_last_ct[j]
                    each_cluster_number[j] = this_last_ct_count[j]
            #此种情况一般不会发生，若发生，表明s_lamdba设置过小
            elif new_part_it_number > 0 and len(new_part_lt) == 0:
                #直接将上一片的主题作为本片的主题
                #每片求均值
                for j in range(len(this_last_ct)):
                    center_topic[j] = this_last_ct[j]
                    each_cluster_number[j] = this_last_ct_count[j]
                
                center_topic = center_topic[0 : len(this_last_ct), :]
                each_cluster_number = each_cluster_number[0 : len(this_last_ct)]
                cluster_tag = cluster_tag[0 : len(this_last_ct)]
                
            elif new_part_it_number == 0 and len(new_part_lt) > 0:    
                #替换一个中心主题
                
                new_part_ct = np.zeros((1, len(PHAI[0])))
                for j in range(len(new_part_lt)):
                    new_part_ct += new_part_lt[j]
                
                new_part_ct = new_part_ct / len(new_part_lt)
                
                min_index = np.argmin(this_last_ct_count)
                
                #找出被删去的主题与哪一个最为相近，合并之
                merge_si = np.zeros(len(this_last_ct), float)
                for j in range(len(this_last_ct)):
                    if j == min_index:
                        merge_si[j] = -1
                    else:
                        merge_si[j] = 1.0 / (SKLD(this_last_ct[min_index], this_last_ct[j]) + 1.0)
                
                merge_des = np.argmax(merge_si)
                
                this_last_ct[min_index] = new_part_ct
                this_last_ct_count[min_index] = len(new_part_lt)
                this_last_ct[merge_des] = (this_last_ct[merge_des] + this_last_ct[min_index]) / 2.0
                #聚类元素个数相加
                this_last_ct_count[merge_des] = this_last_ct_count[merge_des] + this_last_ct_count[min_index]
                
                for j in range(len(this_last_ct)):
                    center_topic[j] = this_last_ct[j]
                    each_cluster_number[j] = this_last_ct_count[j]
                
                for j in range(len(cluster_tag)):
                    #-1变为min_index
                    #min_index变为merge_des
                    if cluster_tag[j] == -1:
                        cluster_tag[j] = min_index
                    elif cluster_tag[j] == min_index:
                        cluster_tag[j] = merge_des
            else:
                #更新前面部分
                for j in range(len(this_last_ct)):
                    center_topic[j] = this_last_ct[j]
                    each_cluster_number[j] = this_last_ct_count[j]
                
                #新增1个主题
                if new_part_it_number == 1:
                    new_part_ct = np.zeros((1, len(PHAI[0])))
                    for j in range(len(new_part_lt)):
                        new_part_ct += new_part_lt[j]
                
                    new_part_ct = new_part_ct / len(new_part_lt)

                    center_topic[-1] = new_part_ct
                    each_cluster_number[-1] = len(new_part_lt)
                    for j in range(len(cluster_tag)):
                        if cluster_tag[j] == -1:
                            cluster_tag[j] = cluster_number - 1
                
                #这里可能会有异常                
                #elif len(new_part_lt) == 1:
                    
                #新增若干个主题   
                else:
                    #谱聚类
                    #print new_part_lt
                    sp_label = spectral_cluster(new_part_lt, new_part_it_number)
                    new_part_ct = np.zeros((new_part_it_number, len(PHAI[0])))
                    new_part_ct_number = np.zeros(new_part_it_number, int)
                    for j in range(len(sp_label)):
                        new_part_ct[sp_label[j]] += new_part_lt[j]
                        new_part_ct_number[sp_label[j]] += 1
                    
                    for j in range(new_part_it_number):
                        new_part_ct[j] = new_part_ct[j] / new_part_ct_number[j]
                        center_topic[len(this_last_ct) + j] = new_part_ct[j]
                        each_cluster_number[len(this_last_ct) + j] = new_part_ct_number[j]
                    
                    new_count = 0
                    for j in range(len(cluster_tag)):
                        if cluster_tag[j] == -1:
                            cluster_tag[j] = cluster_number - new_part_it_number + sp_label[new_count]
                            new_count += 1
                
            #计算文档-主题相似度
            weibo_topic_similarity = np.zeros((cluster_number, len(THETA)))
            THETA = THETA.transpose()
            
            for j in range(len(cluster_tag)):
                weibo_topic_similarity[cluster_tag[j]] += THETA[j]
        
            weibo_topic_similarity = weibo_topic_similarity.transpose()

        run_time.append(str(time.clock() - start))
        print "This time:", str(time.clock() - start)
        # 公共部分
        
        # 加入时间窗口
        ecn_to_string = [str(x) for x in each_cluster_number]    
        ct_window.append(center_topic)
        ct_num_window.append(each_cluster_number)
        ct_wordlist_window.append(this_word_list)
        
        #删除最历史数据
        if len(ct_window) > q:
            ct_window.remove(ct_window[0])
            ct_num_window.remove(ct_num_window[0])
            ct_wordlist_window.remove(ct_wordlist_window[0])

        
        write_matrix_to_text(weibo_topic_similarity, write_directory1 + '/' + str(i + 1) + '.txt')
        write_matrix_to_text(center_topic, write_directory2 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(ecn_to_string, write_directory3 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)
    
    quick_write_list_to_text(run_time, write_filename)


if __name__ == '__main__':
    #start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset/CTLDA/doc_topic'
    read_directory2 = root_directory + u'dataset/CTLDA/topic_word'
    read_directory3 = root_directory + u'dataset/text_model/select_words'
    write_directory1 = root_directory + u'dataset/CTLDA/stream_doc_ct'
    write_directory2 = root_directory + u'dataset/CTLDA/stream_ct_word'
    write_directory3 = root_directory + u'dataset/CTLDA/stream_ct_number'
    write_filename = root_directory + u'dataset/CTLDA/stream_time.txt'


    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    if (not(os.path.exists(write_directory3))):
        os.mkdir(write_directory3)

    
    stream_CT_LDA(read_directory1, read_directory2, read_directory3, write_directory1, write_directory2, write_directory3, write_filename)
    
    #print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
