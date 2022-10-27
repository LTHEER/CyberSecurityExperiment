import numpy as npfrom sklearn.decomposition import PCAclass PCA_Recon_Erro:    """Implementation of anomaly detection base on LinearPCA reconstruction error."""    def __init__(self, matrix, contamination=0.12, random_state=2018):        """        Parameters        --------------------------        - matrix : data set, shape = [n_samples, n_features].        - contamination : float, should be in the range [0, 0.5], default=0.005              The amount of contamination of the data set, i.e. the proportion of outliers in the data set.              Used when fitting to define the threshold on the scores of the samples.        """        self.contamination = contamination        self.random_state = random_state        self.matrix = matrix    def get_ev_ratio(self):        pca = PCA(n_components=None, random_state=self.random_state)        pca_result = pca.fit_transform(self.matrix)        eigenvalues = pca.explained_variance_        # ev_ratio is the cumulative proportion of eigenvalues and the weight of        # reconstruction error corresponding to different number of principal components        ev_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)        return ev_ratio    # using different numbers of principal components to generate a series of reconstruction matrices    def reconstruct_matrix(self):        # the parameter recon_pc_num is the number of top principal components used in the reconstruction matrix.        def reconstruct(recon_pc_num):            pca_recon = PCA(n_components=recon_pc_num, random_state=self.random_state)            pca_reduction = pca_recon.fit_transform(self.matrix)            recon_matrix = pca_recon.inverse_transform(pca_reduction)            assert_description = 'The shape of the reconstruction matrix should be equal to that of the initial matrix.'            assert recon_matrix.shape == self.matrix.shape, assert_description            return recon_matrix        # generating a series of reconstruction matrices        col = self.matrix.shape[1]        recon_matrices = [reconstruct(i) for i in range(1, col + 1)]        # randomly select two reconstruction matrices to verify that they are different        i, j = np.random.choice(range(col), size=2, replace=False)        description = 'The reconstruction matrices generated by different number of principal components are different.'        assert not np.all(recon_matrices[i] == recon_matrices[j]), description        return recon_matrices    # calculate the final anomaly score    def get_anomaly_score(self):        # calculate the modulus of a vector        def compute_vector_length(vector):            square_sum = np.square(vector).sum()            return np.sqrt(square_sum)        # calculate the anomaly score generated by a single reconstruction matrix for all samples        def compute_sub_score(recon_matrix, ev):            delta_matrix = self.matrix - recon_matrix            score = np.apply_along_axis(compute_vector_length, axis=1, arr=delta_matrix) * ev            return score        ev_ratio = self.get_ev_ratio()        reconstruct_matrices = self.reconstruct_matrix()        # summarize the anomaly scores generated by all reconstruction matrices        anomaly_scores = list(map(compute_sub_score, reconstruct_matrices, ev_ratio))        return np.sum(anomaly_scores, axis=0)    # returns indices with the highest anomaly score based on a specific contamination    def get_anomaly_indices(self):        indices_desc = np.argsort(-self.get_anomaly_score())        anomaly_num = int(np.ceil(len(self.matrix) * self.contamination))        anomaly_indices = indices_desc[:anomaly_num]        return anomaly_indices    # returns 1 if the prediction is an anomaly, otherwise returns 0    def predict(self):        anomaly_indices = self.get_anomaly_indices()        pred_result = np.isin(range(len(self.matrix)), anomaly_indices).astype(int)        return pred_result