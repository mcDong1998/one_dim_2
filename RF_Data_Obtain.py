from SVM_RF_classify import *
from Data_Preparation import *
import pandas as pd
import numpy as np


TP = []
TN = []
FP = []
FN = []

# 随机试验10次得平均结果

for i in range(10):
    x_train, x_test, y_train_v, y_test_v, x_verify, y_verify = data_prepare(2018+i)  # 保证每次的数据采样都是不同的
    prediction = RF_grid_train(x_train, y_train_v, x_test, y_test_v)  # 获取模型的预测结果
    # print(prediction)
    # 混淆矩阵数据保存在列表中
    tp, tn, fp, fn = conf_count(prediction, y_test_v)  # 获取混淆矩阵数据
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    # ROC图形将直接保存在本项目路径之下
    print_roc(prediction, y_test_v, 'RF' + str(i))  # 保存ROC图像

print(TP)
print(TN)
print(FP)
print(FN)

# this is about how to save the data in a excel:
writer = pd.ExcelWriter(r'E:\Pycharm_data\pythonProject2\RF_data.xlsx', index=None)
conf = np.zeros((10, 4))
conf[:, 0] = TP
conf[:, 1] = TN
conf[:, 2] = FP
conf[:, 3] = FN
df = pd.DataFrame(conf)
df.to_excel(writer)
writer.save()
