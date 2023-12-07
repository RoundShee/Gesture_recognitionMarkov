import xml.etree.ElementTree as ET
import numpy as np
import pickle

"""
    现在将每个元音字母按时间均匀采集20个点，每组元音字符的20*5=100作为1组存为数据
    数据格式：[ [ax1,ay1], [ax2, ay2] ... [ax20, ay20],
              [ex1,ey1], .... [ux20, uy20]
            ]
"""
name = ['./project1-data/a.xml',
        './project1-data/e.xml',
        './project1-data/i.xml',
        './project1-data/o.xml',
        './project1-data/u.xml']

# for n in name:
#     tree = ET.parse(n)
#     root = tree.getroot()
#     for example in root.findall('trainingExample'):
#         x_raw, y_raw = [], []
#         for dots in example.findall('coord'):
#             x_raw.append(float(dots.get('x')))
#             y_raw.append(float(dots.get('y')))


def get_xy_from_xml(filename, i, fs):
    """
    从filename的xml文件获取第i个xy坐标采样fs的序列
    :param filename:
    :param i:
    :param fs:
    :return:
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    var_i = 0
    x, y = [], []   # 即将return
    for example in root.findall('trainingExample'):
        if var_i == i:  # 找到对应i
            x_raw, y_raw = [], []   # 全部保存
            for dots in example.findall('coord'):
                x_raw.append(float(dots.get('x')))
                y_raw.append(float(dots.get('y')))
            for var_j in range(0,len(x_raw)//fs*fs,len(x_raw)//fs):
                x.append(x_raw[var_j])
                y.append(y_raw[var_j])
        else:
            var_i += 1
    return x, y


def get_one_aeiou_xy_series(start, end):
    """

    :param start:
    :param end:
    :return:
    """
