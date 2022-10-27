import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import sys

sys.path.append('../model/')
import os


def Eif_model_test(datasetpath, modelpath, Frac, Abnormal_proportion):
    # print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    url = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/model/eif.model'
    # print(url)
    if modelpath == None:
        eif = joblib.load(url)
    else:
        eif = joblib.load(modelpath)

    # raw_data_filename1 = "data/NSL_KDD/NSLKDD_total_blance_del.csv"
    raw_data_filename1 = datasetpath
    print("Loading raw data...")
    raw_data1 = pd.read_csv(raw_data_filename1, header=None, low_memory=False)
    # raw_data1 = raw_data1.sample(frac=0.1)
    raw_data1 = raw_data1.sample(frac=Frac)

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

    S = eif.compute_paths(X_in=X_train)


    x = range(len(S))
    y = S

    plt.plot(x, y, "g", marker='D')
    plt.xlabel("sample")
    plt.ylabel("abnormal score")

    plt.title("The abnormal score of each sample")
    plt.legend(loc="lower right")
    plt.savefig('../image/eiftest.png')
    # plt.show()
    s = np.argsort(S)
    print(S)
    print(s)

    # 异常占比
    y_pred = [0 for _ in range(len(s))]
    for y in s[-int(len(s) * Abnormal_proportion):]:
        y_pred[y] = 1

    print(y_pred)

    acc = accuracy_score(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred)

    print("准确率： ", acc)
    print("AUC:", auc_score)


if __name__ == '__main__':
    Eif_model_test("/Users/luolang/CyberSecurityExperiment/data/NSL_KDD/KDDTest+_blance_del.csv", None, 0.2, 0.1)
