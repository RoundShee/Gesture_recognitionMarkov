from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pickle
from make_data import get_one_aeiou_xy_series, plot_aeiou_train_sample, plot_data_from_xml
from hmmlearn import hmm


def view_dbscan_results(start=0, end=1, fs=15, eps=30, min_samples=1):
    """
    查看当前参数设置下,DBSCAN聚类的效果   前期代码整合，十分消耗资源
    :param start: 取样本点开始条目start = 0
    :param end: 结束条目,不包含
    :param fs: 取样点,fs=-1时获得全部
    :param eps: DBSCAN中epsilon
    :param min_samples: DBSCAN最小点数
    :return: plt.show()
    """
    # 左图
    name = ['./project1-data/a.xml', './project1-data/e.xml', './project1-data/i.xml', './project1-data/o.xml',
            './project1-data/u.xml']
    total = []  # 每一项对应0 [x] 1[y]
    for i in range(start, end):
        for j in name:
            total.append(plot_data_from_xml(filename=j, i=i, fs=fs))
    plt.subplot(1, 2, 1)
    for x, y in total:
        plt.scatter(x, y)
    # 获取聚类、连续值
    values = get_one_aeiou_xy_series(start=start, end=end, fs=fs)
    X = np.array(values)

    # # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    # 获取x、y值
    x_all, y_all = plot_aeiou_train_sample(start=start, end=end, fs=fs)
    colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    plt.subplot(1, 2, 2)
    colors_series = []
    for i in labels:
        colors_series.append(colors[int(i)%8])
    plt.scatter(x_all, y_all, c=colors_series)
    plt.show()


def generate_five_model(fs=15, eps=30, min_samples=1, n_components=5, output_filename="./outcome/model_aeiou.pkl"):
    """
    使用训练数据生成对应参数的model组合
    :param output_filename: 输出模型存放位置
    :param eps: DBSCAN方法的epsilon
    :param min_samples: DBSCAN方法的最小样本点
    :param fs: 数据选取-时间尺度上的采样点数
    :param n_components: HMM的隐藏状态数目
    :return: 无,保存到./outcome/model_aeiou.pkl模型列表
    """
    # 每个数据共40个，奇数训练0，2，4，6.   偶数测试 1 ，3， 5
    # 建立-5个模型
    model_aeiou = []
    for i in range(0, 5):
        model_aeiou.append(hmm.GaussianHMM(n_components, n_iter=100))
    # 全部训练输入带入: 0,2,4,....,38
    labels_aeiou = [
        [],     # just a
        [],     # just e
        [],
        [],
        []      # just u
    ]
    for i in range(0, 40, 2):
        start = i
        end = i+1
        values = get_one_aeiou_xy_series(start, end, fs)
        X = np.array(values)
        # 使用DBSCAN进行聚类
        dbscan = DBSCAN(eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        for j in range(0, 5):
            labels_aeiou[j].extend(labels[fs*j:fs*(j+1)])
    obs_aeiou = []
    for j in range(0, 5):
        obs_aeiou.append(np.array(labels_aeiou[j]).reshape(-1, 1))
    i = 0
    for model in model_aeiou:
        model.fit(obs_aeiou[i])
        i += 1
    with open(output_filename, "wb") as file:
        pickle.dump(model_aeiou, file)


def confusion_matrix(filename='./outcome/model_aeiou.pkl', fs=15, eps=30, min_samples=1):
    """
    ~~计算模型的混淆矩阵~~ 只是单个分类器的随便测试
    :param filename: 模型选哪个
    :param fs:
    :param eps:
    :param min_samples:
    :return:
    """
    # 先读取model-一定是aeiou
    with open(filename, "rb") as file:
        model_aeiou = pickle.load(file)
    # 计算矩阵初始化
    matrix = [
        [0, 0, 0, 0, 0],    # 真实a统计结果
        [0, 0, 0, 0, 0],    # 真实e统计结果
        [0, 0, 0, 0, 0],    # 真实i统计结果
        [0, 0, 0, 0, 0],    # 真实o统计结果
        [0, 0, 0, 0, 0],    # 真实u统计结果
    ]
    # 测试数据一次来一组aeiou进行聚类处理
    for i in range(1, 40, 2):       # 测试数据1,3,5,7...,39
        # 读取一组数据
        start = i
        end = i + 1
        values = get_one_aeiou_xy_series(start, end, fs)
        X = np.array(values)
        # 使用DBSCAN进行聚类
        dbscan = DBSCAN(eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        for j in range(0,5):    # 遍历真实aeiou
            real_labels = labels[j*fs:(j+1)*fs]
            pre_labels = judge_aeiou(real_labels, model_aeiou)
            matrix[j][pre_labels] += 1
    print(matrix)


def judge_aeiou(labels, model_file_loaded):
    """
    使用模型集合判断输入的标签序列是哪一个字母
    :param labels: 样本的分类标签序列
    :param model_file_loaded: 已经读取的模型序列
    :return: 返回01234之一
    """
    likelihoods = []
    for model in model_file_loaded:
        likelihoods.append(model.score(np.array(labels).reshape(-1, 1)))
    return likelihoods.index(max(likelihoods))


# generate_five_model()
# confusion_matrix()
