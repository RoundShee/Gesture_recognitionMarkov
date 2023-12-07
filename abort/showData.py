import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from cluster import DBSCAN

# 读取数据
name = ['../project1-data/a.xml', '../project1-data/e.xml', '../project1-data/i.xml', '../project1-data/o.xml', '../project1-data/u.xml']
values = []
for n in name:
    tree = ET.parse(n)
    root = tree.getroot()
    for example in root.findall('trainingExample'):
        x, y = [], []
        for dots in example.findall('coord'):
            x.append(float(dots.get('x')))
            y.append(float(dots.get('y')))
            values.append([float(dots.get('x')), float(dots.get('y'))])
        plt.subplot(1, 2, 1)
        plt.scatter(x, y)
        break

# plt.ion()  # show完不需关闭，程序自动向下执行
# plt.show()

# colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
dbscan = DBSCAN(100, 2)
dbscan.sort(values)
xy = dbscan.get_clusters(using_plot=1)
print(len(xy))
for n in xy:
    x, y = n
    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
plt.show()

# x, y = dbscan.get_kernel(using_plot=1)
# plt.subplot(1, 2, 2)
# plt.scatter(x, y)
# plt.show()
