import xml.etree.ElementTree as ET
import numpy as np
import pickle

"""
    现在将每个元音字母按时间均匀采集20个点，每组元音字符的20*5=100作为1组存为数据
    数据格式：[ [ax1,ay1], [ax2, ay2] ... [ax20, ay20],
              [ex1,ey1], .... [ux20, uy20]
            ]
"""


def get_xy_from_xml(filename, i, fs):
    """
    从filename的xml文件获取第i个xy坐标采样fs的序列
    :param filename:
    :param i:
    :param fs:
    :return: xy
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    var_i = 0
    xy = []   # 即将return
    for example in root.findall('trainingExample'):
        if var_i == i:  # 找到对应i
            if fs == -1:
                for dots in example.findall('coord'):
                    xy.append([float(dots.get('x')), float(dots.get('y'))])
            else:
                x_raw, y_raw = [], []   # 全部保存
                for dots in example.findall('coord'):
                    x_raw.append(float(dots.get('x')))
                    y_raw.append(float(dots.get('y')))
                for var_j in range(0,len(x_raw)//fs*fs,len(x_raw)//fs):
                    xy.append([x_raw[var_j], y_raw[var_j]])
            var_i += 1
        else:
            var_i += 1
    return xy


def get_one_aeiou_xy_series(start, end, step=1, fs=15):
    """
    获取aeiou顺序序列
    :param start: 包含从0开始
    :param end: 不包含
    :param step: 步长
    :param fs
    :return: xy
    """
    name = ['./project1-data/a.xml',
            './project1-data/e.xml',
            './project1-data/i.xml',
            './project1-data/o.xml',
            './project1-data/u.xml']
    xy = []     # 即将返回
    for i in range(start, end, step):
        for j in name:
            xy += get_xy_from_xml(j, i, fs)
    return xy


def plot_data_from_xml(filename, i, fs):
    """
    返回用于plot的[x, y]
    :param filename:
    :param i:
    :param fs:
    :return: [x, y]
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    var_i = 0
    x, y = [], []  # 即将return
    for example in root.findall('trainingExample'):
        if var_i == i:  # 找到对应i
            x_raw, y_raw = [], []  # 全部保存
            for dots in example.findall('coord'):
                x_raw.append(float(dots.get('x')))
                y_raw.append(float(dots.get('y')))
            if fs == -1:
                x += x_raw
                y += y_raw
            else:
                for var_j in range(0, len(x_raw) // fs * fs, len(x_raw) // fs):
                    x.append(x_raw[var_j])
                    y.append(y_raw[var_j])
            var_i += 1
        else:
            var_i += 1
    return [x, y]


def plot_aeiou_train_sample(start, end, step=1, fs=15):
    """
    对应get_one_aeiou_xy_series不加以区分的输出序列
    :param start:
    :param end:
    :param step:
    :param fs
    :return:
    """
    name = ['./project1-data/a.xml',
            './project1-data/e.xml',
            './project1-data/i.xml',
            './project1-data/o.xml',
            './project1-data/u.xml']
    x, y = [], []  # 即将返回
    for i in range(start, end, step):
        for j in name:
            x_i, y_i = plot_data_from_xml(j, i, fs)
            x += x_i
            y += y_i
    return [x, y]
