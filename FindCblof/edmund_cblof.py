import random
import pandas as pd
import numpy as np


# 计算每个点到质心的距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,1)) - centroids  # 相减
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values
    # 计算变化量
    changed = newCentroids - centroids
    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    labels = [0]*len(dataSet)
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        labels[i]=j
    # Centroids是包含所有质心的列表，cluster是所聚类集群的列表，labels是每个数据点属于那个集群的标签
    return centroids, cluster, labels


class CBLOF:
    def __init__(self, n_clusters = 30 ,annomy_s=0.87):
        self.n_clusters = n_clusters
        self.annomy_s = annomy_s

    def fit(self,data):
        X = data
        # Centroids是包含所有质心的列表，cluster是所聚类集群的列表，labels是每个数据点属于那个集群的标签
        centroids, cluster, labels = kmeans(list(X),self.n_clusters)

        # 计算每个集群包含多少样本
        cluster_sizes = []
        for index in range(self.n_clusters):
            cluster_sizes.append(len(cluster[index]))

        # df_cluster_size包含每个集群的序号以及相应的样本数量
        df_cluster_sizes = pd.DataFrame()
        df_cluster_sizes['cluster'] = list(range(self.n_clusters))
        df_cluster_sizes['size'] = df_cluster_sizes['cluster'].apply(lambda c:cluster_sizes[c])
        df_cluster_sizes.sort_values(by=['size'], ascending=False, inplace=True)
        print(df_cluster_sizes)

        # 分割大小簇
        small_clusters=[]
        large_clusters=[]
        count=0
        n_intliers=len(X)*0.9
        for _, row in df_cluster_sizes.iterrows():
            count += row['size']
            if count<n_intliers:
                large_clusters.append(row['cluster'])
            else:
                small_clusters.append(row['cluster'])

        #print(large_clusters)
        #print(small_clusters)
        large_cluster_centers = []

        # 保存所有大簇的质心
        for i in large_clusters:
            large_cluster_centers.append(centroids[i])
        #print(large_cluster_centers)

        #计算两点欧式距离
        def get_distance(a,b):
            return np.sqrt(np.sum(np.square(a - b)))

        # 用来计算每个点与最近大簇的质心的距离
        def decision_function(X, labels):
            n=len(labels)
            distances=[]
            for i in range(n):
                p=X[i]
                label = labels[i]
                # 该样本标签是大簇
                if label in large_clusters:
                    center = centroids[label]
                    d=get_distance(p, center)
                #该样本标签不是大簇
                else:
                    d=None
                    # 遍历计算与每个大簇质心的距离
                    for center in large_cluster_centers:
                        d_temp = get_distance(p, center)
                        if d is None:
                            d=d_temp
                        elif d_temp < d:
                            d=d_temp
                distances.append(d)
            distances=np.array(distances)
            return distances

        distances = decision_function(X, labels)

        # 根据参数异常占比确定哪些样本是异常的
        threshold=np.percentile(distances,self.annomy_s*100)
        anomaly_labels = (distances>threshold)*1

        return anomaly_labels
