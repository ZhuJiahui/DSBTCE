#coding:utf-8
'''
Created on 2013年7月10日
Last on 2013年11月13日

@author: ZhuJiahui506
'''
import xlrd
from openpyxl import Workbook

def open_sheet(filename):
    '''
    打开第一张Excel表格
    :param filename:
    '''
    wb = xlrd.open_workbook(filename)
    ws = wb.sheets()[0]
    return ws

def sheet_to_list(sheet):
    '''
    获取Excel表格数据至一个二维列表，按列存储
    :param sheet:
    '''
    result = []
    for i in range(sheet.ncols):
        result.append(get_data_by_column(sheet, i, 1))  # result为一个二维列表
    return result

def get_data_by_column(sheet, col_index, begin_row=1):
    '''
    获取指定列的数据至一个列表
    :param sheet:
    :param col_index:
    :param begin_row:
    '''
    return sheet.col_values(col_index)[begin_row:]  # col_values获取指定列的全部值为一个列表

def write_list_to_excel(data, write_filename):
    '''
    将数据写入Excel表格
    :param data:
    :param write_filename:
    '''
    print 'Write into %s' % write_filename
    wb = Workbook(optimized_write=True);
    ws = wb.create_sheet(0)
    row_number = len(data[0])
    column_number = len(data)
    
    for i in range(row_number):
        write_column = []
        for j in range(column_number):
            to_write = data[j][i]
            # 将时间转为str
#            if i != 0 and j == 2:
#                toWrite = datetime(*xldate_as_tuple(toWrite,0)).strftime("%Y-%m-%d %H:%M:%S") 
            write_column.append(to_write)
        ws.append(write_column)
        
    wb.save(write_filename)
    print 'Complete Writing!!!' 