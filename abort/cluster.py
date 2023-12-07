"""
    参考PPT《模式识别与机器学习 07 聚类》-李祎liyi@dlut.edu.cn  p39
"""
#  下面代码遇到未知问题，未解决   解决办法：直接调用现有算法
import copy


class DBSCAN:
    def __init__(self, eps, min_pts):
        """对象初始化"""
        self.eps = eps
        self.min_pts = min_pts
        self.eps2 = eps ** 2
        self.kernel = []
        self.clusters = []      # 送入X更新
        self.remain = []

    def sort(self, X):
        """聚类样本X"""
        kernel_sample = []  # 核心样本初始化:存储的是样本加序号
        total_sam = len(X)
        for i in range(0, total_sam):
            neighbor = 0  # 邻域内点数
            for j in range(0, total_sam):
                if i == j:
                    pass
                elif self.get_distance2(X[i], X[j]) < self.eps2:
                    neighbor += 1
                else:
                    pass
            if neighbor >= self.min_pts:
                kernel_sample.append([copy.deepcopy(X[i]), i])
        self.kernel = kernel_sample

        clusters = []
        # 更新remain
        remain = []
        for i in range(0, total_sam):
            remain.append([X[i], i])
        # 对所有kernel_sample带入
        if len(kernel_sample):
            for sample in kernel_sample:
                is_saved = 0
                value, num = sample
                for avoid_error in clusters:    # 避免核心再次计入
                    for value_i, num_i in avoid_error:
                        if num_i == num:
                            is_saved = 1
                if not is_saved:
                    cluster = [copy.deepcopy(sample)]
                    # 下面的for存在问题，只会按照循序遍历一次，因此更新出来的cluster无法扩大
                    update_same = 0
                    while not update_same:
                        count_n = len(cluster)
                        count_new = count_n
                        new_remain = []  # 用于更新remain
                        for other in remain:
                            re = self.is_cluster(cluster, other)
                            if re == 0:     # 已经在cluster中
                                pass
                            elif re == 1:   # 可添加
                                cluster.append(other)
                                count_new += 1
                            elif re == 2:
                                new_remain.append(other)
                        if len(remain) == len(new_remain) and count_new == count_n:
                            update_same = 1
                        else:
                            remain = copy.deepcopy(new_remain)         # 更新剩余
                    clusters.append(copy.deepcopy(cluster))    # 已经聚好的类
        else:
            print('参数不当, 未找到核心对象')
            return -1   # eps过小
        self.clusters = clusters
        self.remain = remain

    @staticmethod
    def get_distance2(x, y):
        x1, x2 = x
        y1, y2 = y
        res = (y1 - x1)**2 + (y2 - x2)**2
        return res

    def is_cluster(self, clu, sample):
        value, pos = sample
        for value_i, pos_i in clu:
            if pos_i == pos:    # 已经在簇中不再添加-返回0
                return 0
            elif self.get_distance2(value, value_i) <= self.eps2:
                return 1        # 簇新成员
            else:
                return 2        # 剩余

    def get_clusters(self, using_plot=0):
        """获得已分类簇"""
        if using_plot == 1:
            xy = []
            for cluster in self.clusters:
                x_series = []
                y_series = []
                for [x, y], i in cluster:
                    x_series.append(x)
                    y_series.append(y)
                xy.append([x_series, y_series])
            return xy
        elif using_plot == 0:
            return self.clusters

    def get_remain(self, using_plot=0):
        """获得已分类但未归成的簇"""
        if using_plot == 1:
            xy = []
            for cluster in self.remain:
                x_series = []
                y_series = []
                for [x, y], i in cluster:
                    x_series.append(x)
                    y_series.append(y)
                xy.append([x_series, y_series])
            return xy
        elif using_plot == 0:
            return self.remain

    def get_kernel(self, using_plot=0):
        """返回kernel_sample"""
        if using_plot == 1:
            x_series = []
            y_series = []
            for [x, y], i in self.kernel:
                x_series.append(x)
                y_series.append(y)
            return [x_series, y_series]
        elif using_plot == 0:
            return self.kernel
