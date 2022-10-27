import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt
import sys
sys.path.append('../image/')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from image.PCA import  PCA_Recon_Erro


def run_pca(raw_data_filename1,raw_data_filename2,Frace):
    # raw_data_filename1 = "/Users/luolang/CyberSecurityExperiment/data/NSL_KDD/KDDTest+_blance_del.csv"
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

    print(X_train)

    P = PCA_Recon_Erro(X_train)

    S = P.get_anomaly_indices()
    x = range(len(S))
    y = S
    plt.plot(x, y, "g", marker='D')
    plt.xlabel("sample")
    plt.ylabel("abnormal score")
    plt.title("The abnormal score of each sample")
    plt.legend(loc="lower right")
    plt.savefig('../image/pca.png')
    # plt.show()

    # raw_data_filename2 = '/Users/luolang/CyberSecurityExperiment/model'
    file = raw_data_filename2 + '/pca.model'
    joblib.dump(P, file)

    y_pred = P.predict()
    acc = accuracy_score(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred)

    print("准确率： ", acc)
    print("AUC:", auc_score)


if __name__ == '__main__':
    run_pca("/Users/luolang/CyberSecurityExperiment/data/NSL_KDD/KDDTest+_blance_del.csv",
            "/Users/luolang/CyberSecurityExperiment/model", 0.2)