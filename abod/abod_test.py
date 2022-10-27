# coding=utf-8
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from edmund_abod import FastABOD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# Frace 随机选取数据集大小
def abod_run(raw_data_filename1, raw_data_filename2, Frace, recent_point, Abnormal_proportion):
    # raw_data_filename1 = "data/NSL_KDD/KDDTest+_blance_del.csv"
    print("Loading raw data...")
    raw_data1 = pd.read_csv(raw_data_filename1, header=None, low_memory=False)
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

    A = FastABOD(X_train, k=recent_point)  # k=20
    # print(A)
    # 模型保存路径
    file = raw_data_filename2 + '/abod.model'
    joblib.dump(A, file)

    Afaf = A.fast_abod()
    print(Afaf)
    x = range(len(Afaf))
    y = Afaf

    plt.plot(x, y, "g", marker='D')
    plt.xlabel("sample")
    plt.ylabel("abnormal score")

    plt.title("The abnormal score of each sample")
    plt.legend(loc="lower right")
    plt.savefig('../image/fastabod.png')
    # plt.show()

    a = np.argsort(Afaf)

    # 异常占比 0.12
    y_pred = [0 for _ in range(len(a))]
    for y in a[-int(len(a) * Abnormal_proportion):]:
        y_pred[y] = 1

    print(y_pred)

    acc = accuracy_score(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred)

    print("准确率： ", acc)
    print("AUC:", auc_score)


if __name__ == '__main__':
    abod_run("/Users/luolang/CyberSecurityExperiment/data/NSL_KDD/KDDTest+_blance_del.csv",
             "/Users/luolang/CyberSecurityExperiment/model", 0.4, 40, 0.15)
