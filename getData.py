from hmmlearn import hmm
import numpy as np
import xml.etree.ElementTree as ET
import pickle

# 读取数据
tree = ET.parse('./project1-data/a.xml')
root = tree.getroot()

# 提取奇数样本-提取时间轴上均匀分布的十个点构成样本集
all_sample = []
num = 0
for example in root.findall('trainingExample'):
    if num % 2 == 1:
        sample = []
        for coord in example.findall('coord'):
            sample.append([float(coord.get('x')), float(coord.get('y'))])
        part_sample = []
        for i in range(0, (len(sample)//10)*10, len(sample)//10):
            part_sample.append(sample[i])
        all_sample.append(part_sample)
    else:
        pass
    num += 1

with open("./outcome/even.list", "wb") as file:
    pickle.dump(all_sample, file)
    
# 训练模型
# model = hmm.MultinomialHMM(n_components=10, algorithm='viterbi', n_iter=1000, tol=0.01, params='ste', init_params='ste')
# model.fit(all_sample, 10)
# with open("./outcome/model.pkl", "wb") as file:
#     pickle.dump(model, file)
