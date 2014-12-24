# -*- coding: utf-8 -*-
'''
Created on 2013年11月14日

@author: ZhuJiahui506
'''

import numpy as np

def get_text_to_single_list(data_list, read_filename):
    '''
    Read txt file to an 1-dimension list.
    No return.
    
    :param data_list:
    :param read_filename:
    '''
    f = open(read_filename, 'rb')
    data_list2 = f.readlines()
    f.close()
    for each in data_list2:
        data_list.append(each.strip())
    
def get_text_to_single_list2(read_filename):
    '''
    Read txt file to an 1-dimension list.
    With return.
    
    :param read_filename:
    
    output data_list: the result list.
    '''
    f = open(read_filename, 'rb')
    data_list = f.readlines()
    f.close()
    data_list = [each.strip() for each in data_list]
    return data_list

def get_text_to_complex_list(data_list, read_filename, start_column_index):
    '''
    Read txt file to a 2-dimension list.
    No return.
    
    :param data_list:
    :param read_filename:
    :param start_column_index: 开始列的索引
    '''
    f = open(read_filename, 'rb')
    line = f.readline()
    while line:
        data_list.append(line.split()[start_column_index:])
        line = f.readline()
    f.close()
    
def get_text_to_complex_list2(data_list, read_filename, start_column_index, end_column_index):
    '''
    Read txt file to a 2-dimension list.
    No return.
    
    :param data_list:
    :param read_filename:
    :param start_column_index: 开始列的索引
    :param end_column_index: 结束列的索引
    '''
    f = open(read_filename, 'rb')
    line = f.readline()
    while line:
        data_list.append(line.split()[start_column_index : end_column_index])
        line = f.readline()
        
    f.close()
    
def get_text_to_nparray(read_filename, data_type='float'):
    '''
    Read txt file to a 2-dimension numpy array.
    With return.
    
    :param read_filename:
    :param data_type: 设定读取的结果类型 float或int
    
    output data_list: the result 2-dimension numpy array.
    '''
    f = open(read_filename, 'r')
    data = f.readlines()
    f.close()
    data_list2 = []
    if data_type == 'int':
        for each in data:
            data_list2.append([int(x) for x in each.split()])
        data_list = np.array(data_list2)
    elif data_type == 'float':
        for each in data:
            data_list2.append([float(x) for x in each.split()]) 
        data_list = np.array(data_list2)
    else:
        data_list = np.array([0])
        print 'Type Error!!!'
    
    return data_list


def write_list_to_text(data_list, write_filename):
    '''
    Write 1-dimension list to txt.
    No return.
    For big data.
    
    :param data_list:
    :param write_filename:
    '''
    f = open(write_filename, 'w')
    
    for i in range(len(data_list) - 1):
        f.write(str(data_list[i])) 
        f.write('\n')
    
    f.write(str(data_list[len(data_list) - 1])) 
        
    f.close()
    
def quick_write_list_to_text(data_list, write_filename):
    '''
    Write 1-dimension list to txt.
    No return.
    For small data.
    
    :param data_list:
    :param write_filename:
    '''
    f = open(write_filename, 'w')
    f.writelines("\n".join(data_list))   
    f.close()
    
def quick_write_list_to_text2(data_list, write_filename):
    '''
    Already with "\n" join in the list,like ['ZhuJiahui\n', 'HuangJiajia\n'].
    No return.
    
    :param data_list:
    :param write_filename:
    '''
    f = open(write_filename, 'w') 
    f.writelines(data_list)   
    f.close()
    
def write_list_to_text_by_column(data_list, write_filename):
    '''
    Write 2-dimension list to txt by column.
    No return.
    For big data.
    
    :param data_list:
    :param write_filename:
    '''
    f = open(write_filename, 'w')
    
    row = len(data_list[0])
    column = len(data_list)
    for i in range(row - 1):
        for j in range(column - 1):
            f.write(str(data_list[j][i])) 
            f.write(' ')
        f.write(str(data_list[column - 1][i]))
        f.write('\n')
    
    for k in range(column - 1):
        f.write(str(data_list[k][row - 1]))
        f.write(' ') 
    
    f.write(str(data_list[column - 1][row - 1]))  
    f.close()

def write_list_to_text_by_row(data_list, write_filename):
    '''
    Write 2-dimension list to txt by row.
    No return.
    For big data.
    
    :param data_list:
    :param write_filename:
    '''
    
    f = open(write_filename, 'w')
    
    row = len(data_list)
    for i in range(row - 1):
        each_row = [str(each) for each in data_list[i]]
        f.write(" ".join(each_row))
        f.write('\n')
    
    last_row = [str(each) for each in data_list[row - 1]]
    f.write(" ".join(last_row))  
    f.close()


def write_matrix_to_text(data_list, write_filename):
    '''
    Write matrix to txt. The matrix may be 1 or 2-dimension list or numpy array.
    No return.

    :param data_list:
    :param write_filename:
    '''
    data = []
    for each in data_list:
        data.append(" ".join([str(x) for x in each]))
    quick_write_list_to_text(data, write_filename)


if __name__ == '__main__':
    pass