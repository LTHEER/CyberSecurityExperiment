import numpy as np

class PCAcomponent():
    def __init__(self, X, N):
        self.X = X
        self.N = N
        self.variance_ratio = []
        self.low_dataMat = []
        self.reconMat = []
        self.eigenvalues = []

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        # 若rowvar非0，一列代表一个样本；为0，一行代表一个样本
        covMat = np.cov(dataMat, rowvar=False)
        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(self.N + 1):-1]  # 取前N个较大的特征值
        small_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
        self.low_dataMat = dataMat * small_eigVect  # 投影变换后的新矩阵
        self.reconMat = (self.low_dataMat * small_eigVect.I) + X_mean  # 重构数据
        # 输出每个维度所占的方差百分比
        self.eigenvalues = [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]

    def fit(self):
        self._fit()
        return self

class PCA_Recon_Error:

    def __init__(self, matrix, contamination=0.12):
        """
        参数
        - matrix : 数据集, shape = [n_samples, n_features].
        - contamination : 异常占比，默认为0.12
        """
        self.matrix = matrix
        self.contamination = contamination

    def get_ev_ratio(self):
        pca = PCAcomponent(self.matrix,min(self.matrix.shape))
        eigenvalues = pca.eigenvalues
        # ev_ratio 是特征值与不同主成分数对应的重构误差权重的累积比例
        ev_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return ev_ratio

    # 利用不同数量的主成分生成一系列重构矩阵
    def reconstruct_matrix(self):
        # 参数 recon_pc_num 是重建矩阵中使用的顶部主成分的数量
        def reconstruct(recon_pc_num):
            pca_recon = PCAcomponent(self.matrix,N=recon_pc_num)
            pca_recon.fit()
            # pca_reduction = pca_recon.low_dataMat
            recon_matrix = pca_recon.reconMat
            return recon_matrix

        # 生成一系列重构矩阵
        col = self.matrix.shape[1]
        recon_matrices = [reconstruct(i) for i in range(1, col + 1)]

        # 随机选择两个重建矩阵以验证它们是否不同
        i, j = np.random.choice(range(col), size=2, replace=False)
        description = 'The reconstruction matrices generated by different number of principal components are different.'
        assert not np.all(recon_matrices[i] == recon_matrices[j]), description
        return recon_matrices

    # 计算最终异常得分
    def get_anomaly_score(self):
        # 计算向量的模
        def compute_vector_length(vector):
            square_sum = np.square(vector).sum()
            return np.sqrt(square_sum)

        # 计算所有样本的单个重建矩阵生成的异常得分
        def compute_sub_score(recon_matrix, ev):
            delta_matrix = self.matrix - recon_matrix
            score = np.apply_along_axis(compute_vector_length, axis=1, arr=delta_matrix) * ev
            return score

        ev_ratio = self.get_ev_ratio()
        reconstruct_matrices = self.reconstruct_matrix()
        # 汇总所有重建矩阵生成的异常得分
        anomaly_scores = list(map(compute_sub_score, reconstruct_matrices, ev_ratio))
        return np.sum(anomaly_scores, axis=0)

    # 返回基于特定异常占比的异常得分最高的指数
    def get_anomaly_indices(self):
        indices_desc = np.argsort(-self.get_anomaly_score())
        anomaly_num = int(np.ceil(len(self.matrix) * self.contamination))
        anomaly_indices = indices_desc[:anomaly_num]
        return anomaly_indices

    # 如果预测是异常，则返回1，否则返回0
    def predict(self):
        anomaly_indices = self.get_anomaly_indices()
        pred_result = np.isin(range(len(self.matrix)), anomaly_indices).astype(int)
        return pred_result