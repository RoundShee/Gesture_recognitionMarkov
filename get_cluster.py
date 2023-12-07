from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# 生成一些随机的二维数据点
name = ['./project1-data/a.xml', './project1-data/e.xml', './project1-data/i.xml', './project1-data/o.xml', './project1-data/u.xml']
values = []
x_all = []
y_all = []

for n in name:
    tree = ET.parse(n)
    root = tree.getroot()
    for example in root.findall('trainingExample'):
        x, y = [], []
        for dots in example.findall('coord'):
            x.append(float(dots.get('x')))
            y.append(float(dots.get('y')))
            values.append([float(dots.get('x')), float(dots.get('y'))])
            x_all.append(float(dots.get('x')))
            y_all.append(float(dots.get('y')))
        plt.subplot(1, 2, 1)
        plt.scatter(x, y)
        break
X = np.array(values)

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=20, min_samples=1)
labels = dbscan.fit_predict(X)

# 数据长度
print(len(x_all))
# 看看图
colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
colors_series = []
plt.subplot(1, 2, 2)
for i in labels:
    colors_series.append(colors[int(i)%8])
plt.scatter(x_all, y_all, c=colors_series)
plt.show()



