import numpy as np

class FastABOD:
    def __init__(self, D, k=5):
        # D是数据集
        # k是考虑最近多少个点，默认为5，作为参数用来计算与附近点的所有角度组合的方差
        self.D = D
        self.k = k

    def get_dist(self):
        # 计算欧式距离矩阵
        # 因为 欧式距离矩阵dists = sqrt（X*X + Y*Y - 2*X*Y.T）
        # X=Y 所以dists = sqrt（2*X*X - 2*X*X.T）
        d1 = 2 * np.sum(np.square(self.D), axis=1)
        d2 = -2 * np.dot(self.D, self.D.T)
        dists = np.sqrt(d1 + d2)
        return dists

    def NearestNeighbors(self, arr):
        # 找到最近k个点
        # 距离排序，选取前K个，返回最近k的点
        inds_sort = np.argsort(arr)
        neighbor_ind = inds_sort[1:self.k + 1]  # 邻域内点索引
        return neighbor_ind

    def calculate_weight(self,pivot, n_1, n_2):
        # 实现1/(||AB||*||AC||)
        diff_AB = np.subtract(n_1,pivot)
        diff_AC = np.subtract(n_2,pivot)
        return 1.0/(np.linalg.norm(diff_AB)*np.linalg.norm(diff_AC))

    def calculate_angle(self,pivot, n_1, n_2):
        # 实现<AB,AC>/(||AB||^2*||AC||^2)
        diff_AB = np.subtract(n_1,pivot)
        diff_AC = np.subtract(n_2,pivot)
        return np.dot(diff_AB,diff_AC)/(np.linalg.norm(diff_AB)**2 * np.linalg.norm(diff_AC)**2)

    def fast_abod(self):
        '''
           返回：带有附加ABOF因子列的i/p数据集
        '''
        # 初始化ABV阵列
        D = self.D
        sigma_theta = []

        # 检查数据集，判断是不是n维数组
        if type(D) != np.ndarray:
            D = np.asarray(D)

        # 找k个最近的邻居
        dist = self.get_dist()
        neighbors = []
        for d in dist:
            neighbors.append(self.NearestNeighbors(d))

        # 迭代点
        for i in range(len(D)):
            local_theta_n_0 = 0.0
            local_theta_n_1 = 0.0
            local_theta_d = 0.0
            local_N = neighbors[i]
            for j in range(len(local_N)):
                for k in range(j + 1, len(local_N)):
                    # 分别实现公式里的三部分
                    local_theta_n_0 += (self.calculate_weight(D[i], D[local_N[j]], D[local_N[k]]) * \
                                       self.calculate_angle(D[i], D[local_N[j]], D[local_N[k]])) ** 2
                    local_theta_n_1 += self.calculate_weight(D[i], D[local_N[j]], D[local_N[k]]) * \
                                       self.calculate_angle(D[i],D[local_N[j]],D[local_N[k]])
                    local_theta_d += self.calculate_weight(D[i], D[local_N[j]], D[local_N[k]])
            # 考虑分母为0的情况
            if local_theta_d == 0:
                sigma_theta.append(0)
            else:
                # 计算ABOD因子
                sigma_theta.append((local_theta_n_0 / local_theta_d) - (local_theta_n_1 / local_theta_d) ** 2)

        # 返回ABOD因子
        return sigma_theta