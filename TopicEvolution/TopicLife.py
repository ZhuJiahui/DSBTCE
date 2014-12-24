# -*- coding: utf-8 -*-
'''
Created on 2014年5月14日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import get_text_to_single_list, write_matrix_to_text
import time
from KLD import SKLD
from MergeAllCenter import merge_all_center
from operator import itemgetter

'''
Step 10
Central Topic LDA.
'''

def topic_life(read_directory1, read_directory2, read_directory3, write_directory1):
    
    gamma = 0.65
    delta = 0.80
    
    #file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    q = 4
    start_batch = 46
    interval = 7
    end_batch = start_batch + interval
    
    all_topic_batch, new_word_list, all_count = merge_all_center(read_directory1, read_directory2, start_batch, end_batch)
    
    evolution_matrix = np.zeros((all_count, all_count), int)
    
    previous_topics = []
    previous_num = []
    previous_intensity = []
    
    start_index = 0
    end_index = 0
    
    for i in range(len(all_topic_batch)):
        this_topic_intensity = []
        get_text_to_single_list(this_topic_intensity, read_directory3 + '/' + str(start_batch + i) + '.txt')
        this_topic_intensity = [int(x) for x in this_topic_intensity]
        print this_topic_intensity
        
        if i == 0:
            for j in range(len(all_topic_batch[i])):
                evolution_matrix[j, j] = 1
                previous_topics.append(all_topic_batch[i][j])
                previous_intensity.append(this_topic_intensity[j])
            
            start_index = 0
            end_index += len(all_topic_batch[i])
            
            previous_num.append(len(all_topic_batch[i]))

        else:
            kl_matrix = np.zeros((len(all_topic_batch[i]), len(previous_topics)))
            
            for j in range(len(all_topic_batch[i])):
                for k in range(len(previous_topics)):
                    kl_matrix[j, k] = 1.0 / (SKLD(all_topic_batch[i][j], previous_topics[k]) + 1.0)
            
            #判断出现
            for j in range(len(kl_matrix)):
                #if np.max(kl_matrix[j]) < gamma:
                evolution_matrix[end_index + j, end_index + j] = 1
            
            #判断消失
            for j in range(len(kl_matrix[0])):
                if np.max(kl_matrix[:, j]) < gamma:
                    evolution_matrix[start_index + j, start_index + j] = -1
            
            #判断延续
            for j in range(len(kl_matrix)):
                for k in range(len(kl_matrix[j])):
                    if kl_matrix[j][k] >= delta:
                        evolution_matrix[start_index + k, end_index + j] = 2
                        evolution_matrix[end_index + j, start_index + k] = 2
            
            #判断合并
            for j in range(len(kl_matrix)):
                latent_merge_index = []
                si_value = []
                for k in range(len(kl_matrix[j])):
                    if kl_matrix[j][k] >= gamma and kl_matrix[j][k] < delta:
                        latent_merge_index.append(k)
                        si_value.append(kl_matrix[j][k])
                
                
                
                if len(latent_merge_index) >= 2:
                    sl = zip(latent_merge_index, si_value)
                    sl = sorted(sl, key = itemgetter(1), reverse=True)
                    latent_merge_index = []
                
                    m_count = 0
                    for each in sl:
                        latent_merge_index.append(each[0])
                        m_count += 1
                    
                        if m_count >= 3:
                            break
                    
                    Z = np.zeros(len(all_topic_batch[i][0]))
                    all_intensity = 0
                    for each in latent_merge_index:
                        Z += previous_topics[each] * previous_intensity[each]
                        all_intensity += previous_intensity[each]
                    
                    Z = Z / all_intensity
                    related = 1.0 / (SKLD(all_topic_batch[i][j], Z) + 1.0)
                    
                    if related > delta:
                        for each in latent_merge_index:
                            evolution_matrix[start_index + each, end_index + j] = 3
                            evolution_matrix[end_index + j, start_index + each] = 3
            #判断分裂
            if len(kl_matrix) > 1: 
                for j in range(len(kl_matrix[0])):
                    latent_split_index = []
                    for k in range(len(kl_matrix)):
                        if kl_matrix[k][j] >= gamma and kl_matrix[k][j] < delta:
                            latent_split_index.append(k)
                
                    if len(latent_split_index) >= 2:
                        Z = np.zeros(len(all_topic_batch[i][0]))
                        all_intensity = 0
                        for each in latent_split_index:
                            Z += all_topic_batch[i][each] * this_topic_intensity[each]
                            all_intensity += this_topic_intensity[each]
                    
                        Z = Z / all_intensity
                        related = 1.0 / (SKLD(previous_topics[j], Z) + 1.0)
                    
                        if related > delta:
                            for each in latent_split_index:
                                evolution_matrix[start_index + j, end_index + each] = 4
                                evolution_matrix[end_index + each, start_index + j] = 4     
            
            for j in range(len(all_topic_batch[i])):
                previous_topics.append(all_topic_batch[i][j])
                previous_intensity.append(this_topic_intensity[j])
            
            previous_num.append(len(all_topic_batch[i]))
            
            if len(previous_num) > q:
                start_index += previous_num[0]
                for l in range(previous_num[0]):
                    previous_topics.remove(previous_topics[0])
                    previous_intensity.remove(previous_intensity[0])
                
                previous_num.remove(previous_num[0])
                
            
            end_index += len(all_topic_batch[i])
        
        write_matrix_to_text(evolution_matrix, write_directory1 + '/' + str(i + 1) + '.txt')        
        print "Evolution %d Completed." % (i + 1)
        

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory = root_directory + u'dataset/topic_evolution'
    
    read_directory1 = root_directory + u'dataset/CTLDA/stream_ct_word'
    read_directory2 = root_directory + u'dataset/text_model/select_words'
    read_directory3 = root_directory + u'dataset/CTLDA/stream_ct_number'
    
    write_directory1 = write_directory + u'/evolution_matrix'

    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
        os.mkdir(write_directory1)

    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)

    topic_life(read_directory1, read_directory2, read_directory3, write_directory1)
  
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    