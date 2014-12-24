#coding:utf-8
'''
Created on 2013年7月17日

@author: ZhuJiahui
'''

def time_convert(time):
    """
    将系统时间转化为小时
    :param time:
    :return:
    """
    date = int(time)
    hour = round((time - date) * 24)
    time = date + hour / 24
    return time