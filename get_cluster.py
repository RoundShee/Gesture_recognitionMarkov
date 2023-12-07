from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from make_data import get_one_aeiou_xy_series, plot_aeiou_train_sample
from hmmlearn import hmm

# 样本起始设置
start, end = 0, 1
values = get_one_aeiou_xy_series(start, end, fs=-1)
X = np.array(values)

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=30, min_samples=1)
labels = dbscan.fit_predict(X)
x_all, y_all = plot_aeiou_train_sample(start, end, fs=-1)
colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
colors_series = []
for i in labels:
    colors_series.append(colors[int(i)%8])
plt.scatter(x_all, y_all, c=colors_series)
plt.show()


