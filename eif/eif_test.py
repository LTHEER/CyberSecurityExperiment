# coding=utf-8
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import edmund_eif as iso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import sys

sys.path.append("../data/")


# raw_data_filename1 数据集地址
# Frace 数据集随机采样 默认 0.1
# raw_data_filename2 模型保存地址
# Anomoy 异常占比 默认 0.17
# X(list of list of floats):训练数据,坐标点列表[x1,x2,...,xn]
# ntrees(int):要使用的树的数量 默认500
# sample_size(int):用于创建每棵树的子样本的大小,必须小于X 默认512
# limit(int):允许的最大树深度.默认情况下，该值设置为二叉树中未成功搜索的平均长度。
# ExtensionLevel(int):指定选择用于分割数据的超平面的自由度.必须小于数据集的维度n 默认0


def eif_run(raw_data_filename1, raw_data_filename2, Frace, Anomy, Ntrees, Sample_size, Limit, EXtensionLevel):
    # raw_data_filename1 = "data/NSL_KDD/KDDTest+_blance_del.csv"
    print("Loading raw data...")
    raw_data1 = pd.read_csv(raw_data_filename1, header=None, low_memory=False)
    print("Data loading is complete! ! ! !")

    # raw_data1 = raw_data1.sample(frac=0.1)
    raw_data1 = raw_data1.sample(frac=Frace)

    def lookData(raw_data):
        last_column_index = raw_data.shape[1] - 1
        print("print data labels:")
        print(raw_data[last_column_index].value_counts())

    lookData(raw_data1)
    traindata = raw_data1

    for column in traindata.columns:
        if traindata[column].dtype == type(object):
            le = LabelEncoder()  # 标签编码,即是对不连续的数字或者文本进行编号,转换成连续的数值型变量
            traindata[column] = le.fit_transform(traindata[column])

    X1 = traindata.iloc[:, :traindata.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
    Y1 = traindata.iloc[:, traindata.shape[1] - 1:]

    scaler = Normalizer().fit(X1)
    trainX = scaler.transform(X1)

    X_train = np.array(trainX)
    y_train = np.array(Y1)

    # F = iso.iForest(X_train, ntrees=500, sample_size=512, ExtensionLevel=0)

    F = iso.iForest(X_train, ntrees=Ntrees, sample_size=Sample_size, limit=Limit, ExtensionLevel=EXtensionLevel)
    # file = 'model/eif.model'
    file = raw_data_filename2 + '/eif.model'
    joblib.dump(F, file)

    S = F.compute_paths(X_in=X_train)

    x = range(len(S))
    y = S

    plt.plot(x, y, "g", marker='D')
    plt.xlabel("sample")
    plt.ylabel("abnormal score")

    plt.title("The abnormal score of each sample")
    plt.legend(loc="lower right")
    plt.savefig('../image/eif.png')
    # plt.show()

    s = np.argsort(S)
    print(S)
    print(s)

    # 0.17 异常占比 Anomy
    y_pred = [0 for _ in range(len(s))]
    for y in s[-int(len(s) * Anomy):]:
        y_pred[y] = 1

    print(y_pred)

    acc = accuracy_score(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred)

    print("准确率： ", acc)
    print("AUC:", auc_score)


if __name__ == '__main__':
    eif_run("/Users/luolang/CyberSecurityExperiment/data/NSL_KDD/KDDTest+_blance_del.csv",
            '/Users/luolang/CyberSecurityExperiment/model', 0.1, 0.17, 500, 512, 5, 0)
