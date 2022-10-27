# coding=utf-8
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from LOf.edmund_lof import LOF
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('../image/')

def run_lof_test(raw_data_filename1, raw_data_filename2, Frace, Domain_Index, Abnormal_proportion):
    # raw_data_filename1 = "data/NSL_KDD/KDDTest+_blance_del.csv"
    print("Loading raw data...")
    raw_data1 = pd.read_csv(raw_data_filename1, header=None, low_memory=False)

    def lookData(raw_data):
        last_column_index = raw_data.shape[1] - 1
        print("print data labels:")
        print(raw_data[last_column_index].value_counts())

    raw_data1 = raw_data1.sample(frac=Frace)
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

    # LOf = LOF(X_train, 200, epsilon=1.3)
    lof = LOF(X_train, Domain_Index)
    score = lof.run()

    # file = 'model/LOf.model'
    file = raw_data_filename2 + '/LOf.model'
    joblib.dump(lof, file)

    s = np.argsort(score)

    # Abnormal_proportion = 0.14
    y_pred = [0 for _ in range(len(s))]
    for y in s[-int(len(s) * Abnormal_proportion):]:
        y_pred[y] = 1

    print(y_pred)

    acc = accuracy_score(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred)

    print("准确率： ", acc)
    print("AUC:", auc_score)

    x = range(len(score))
    y = score

    plt.plot(x, y, "g", marker='D')
    plt.xlabel("sample")
    plt.ylabel("abnormal score")

    plt.title("The abnormal score of each sample")
    plt.legend(loc="lower right")
    plt.savefig('../image/LOf.png')
    # # plt.show()


if __name__ == '__main__':
    run_lof_test("/Users/luolang/CyberSecurityExperiment/data/NSL_KDD/KDDTest+_blance_del.csv",
                 '/Users/luolang/CyberSecurityExperiment/model', Frace=0.2,
                 Domain_Index=200, Abnormal_proportion=0.14)
