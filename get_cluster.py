from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pickle
from make_data import get_one_aeiou_xy_series, plot_aeiou_train_sample
from hmmlearn import hmm

#
# # 样本起始设置
# start, end = 1, 2
# fs = 15
# values = get_one_aeiou_xy_series(start, end, fs)
# X = np.array(values)
#
# # 使用DBSCAN进行聚类
# dbscan = DBSCAN(eps=30, min_samples=1)
# labels = dbscan.fit_predict(X)
#
# x_all, y_all = plot_aeiou_train_sample(start, end, fs)
# colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
# colors_series = []
# for i in labels:
#     colors_series.append(colors[int(i)%8])
# plt.scatter(x_all, y_all, c=colors_series)
# plt.show()
# model = hmm.GaussianHMM(n_components=10, algorithm='viterbi', n_iter=1000, tol=0.01, params='ste', init_params='ste')
# model.fit([labels])

# 每个数据共40个，奇数训练0，2，4，6.   偶数测试 1 ，3， 5
# 建立-5个模型
model_aeiou = []
for i in range(0, 5):
    model_aeiou.append(hmm.GaussianHMM(n_components=5, n_iter=100))
# 全部训练输入带入: 0,2,4,....,38
# labels_a = [], labels_e = [], labels_i = [], labels_o = [], labels_u = []
for i in range(0, 40, 2):
    start = i
    end = i+1
    fs = 15     # 每个字形只用15个
    values = get_one_aeiou_xy_series(start, end, fs)
    X = np.array(values)
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=30, min_samples=1)
    labels = dbscan.fit_predict(X)
    # 已知标签
    obs_a = np.atleast_2d(labels[0:15]).T
    obs_e = np.atleast_2d(labels[15:30]).T
    obs_i = np.atleast_2d(labels[30:45]).T
    obs_o = np.atleast_2d(labels[45:60]).T
    obs_u = np.atleast_2d(labels[60:75]).T
    obs_aeiou = [obs_a, obs_e, obs_i, obs_o, obs_u]
    # 分别训练markov
    i = 0
    for model in model_aeiou:
        model.fit(obs_aeiou[i])
        i += 1
with open("./outcome/model_aeiou.pkl", "wb") as file:
    pickle.dump(model_aeiou, file)

