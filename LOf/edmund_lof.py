# coding=utf-8
import numpy as np


class LOF:
    '''
    计算每个点的局部离群因子
    参数
    k(int):邻域包含多少个点
    epsilon(float):判断异常的阈值
    '''

    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.N = self.data.shape[0]

    def get_dist(self):
        # 计算欧式距离矩阵
        # 因为 欧式距离矩阵dists = sqrt（X*X + Y*Y - 2*X*Y.T）
        # X=Y 所以dists = sqrt（2*X*X - 2*X*X.T）
        d1 = 2 * np.sum(np.square(self.data), axis=1)
        d2 = -2 * np.dot(self.data, self.data.T)
        dists = np.sqrt(d1 + d2)
        return dists

    def _kdist(self, arr):
        # 计算k距离
        inds_sort = np.argsort(arr)
        neighbor_ind = inds_sort[1:self.k + 1]  # 邻域内点索引
        return neighbor_ind, arr[neighbor_ind[-1]]

    def get_rdist(self):
        # 计算可达距离
        dist = self.get_dist()
        nei_kdist = np.apply_along_axis(self._kdist, 1, dist)
        nei_inds, kdist = zip(*nei_kdist)
        for i, k in enumerate(kdist):
            ind = np.where(dist[i] < k)  # 实际距离小于k距离，则可达距离为k距离
            dist[i][ind] = k
        return nei_inds, dist

    def get_lrd(self, nei_inds, rdist):
        # 计算局部可达密度
        lrd = np.zeros(self.N)
        for i, inds in enumerate(nei_inds):
            s = 0
            for j in inds:
                s += rdist[j, i]
            lrd[i] = self.k / s
        return lrd

    def run(self):
        # 计算局部离群因子
        nei_inds, rdist = self.get_rdist()
        lrd = self.get_lrd(nei_inds, rdist)
        score = np.zeros(self.N)
        for i, inds in enumerate(nei_inds):
            N = len(inds)
            lrd_nei = sum(lrd[inds])
            score[i] = lrd_nei / self.k / lrd[i]

        return score
