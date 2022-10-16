"""使用SVM和RF对一维数据集进行分类，the func contains:
1. SVM_grid_train()    -->return y_pred_test
2. RF_simple_train()  -->return y_pred_test
3. RF_grid_train() -->return y_pred_test
4. conf_count(y_pred_test, y_test_v)  -->return TP, TN, FP, FN
5. print_roc(y_pred_test, y_test_v) 无返回值,但会保存图像
6. delong_test delong_test(labels, prediction_1, prediction_2) -->return p
"""
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn as sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from Data_Preparation import *
from Delong_test import *

# data
# x_train, x_test, y_train_v, y_test_v, x_verify, y_verify = data_prepare()


# svm_训练函数定义
def SVM_grid_train(x_train, y_train_v, x_test, y_test_v):
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid)
    clf.fit(x_train, y_train_v)
    y_pred_test = clf.predict(x_test)

    print("Test data metrics:")
    print(sklearn.metrics.classification_report(y_true=y_test_v, y_pred=y_pred_test))
    return y_pred_test


# RF 简单 train 定义
def RF_simple_train(x_train, y_train_v, x_test, y_test_v):
    """train 函数定义，需要的数据在外部进行定义"""
    rf = RandomForestClassifier()  # 调用随机森林分类器
    rf.fit(x_train, y_train_v)  # 巡林数据和标签

    y_pred_train = rf.predict(x_train)
    y_pred_test = rf.predict(x_test)
    # 矩阵
    print("Training metrics:")
    print(sklearn.metrics.classification_report(y_true=y_train_v, y_pred=y_pred_train))
    print("Test data metrics:")
    print(sklearn.metrics.classification_report(y_true=y_test_v, y_pred=y_pred_test))

    return y_pred_test


# 使用网络搜索算法进行RF的调优
# =========================== Using Grid Search for hyper parameter tuning ===================================
def RF_grid_train(x_train, y_train_v, x_test, y_test_v):
    rf = RandomForestClassifier()  # 调用随机森林分类器
    clf = GridSearchCV(rf, param_grid={'n_estimators': [100, 200], 'min_samples_leaf': [2, 3]})
    model = clf.fit(x_train, y_train_v)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # training metrics
    print("Training metrics:")
    print(sklearn.metrics.classification_report(y_true=y_train_v, y_pred=y_pred_train))

    # test data metrics
    print("Test data metrics:")
    print(sklearn.metrics.classification_report(y_true=y_test_v, y_pred=y_pred_test))

    return y_pred_test

# 使用聚类以及超参数调优


# 混淆矩阵简单计算方法
def conf_count(y_pred_test, y_test_v):
    a = y_pred_test-y_test_v
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(41):
        if a[i] == 1:
            FN += 1
        if a[i] == -1:
            FP += 1
        TP = 22-FN
        TN = 19-FP

    return TP, TN, FP, FN


# 画ROC曲线
def print_roc(y_pred_test, y_test_v, save_name):
    y_true =y_test_v  # 非二进制需要pos_label
    y_score = y_pred_test
    fpr_0, tpr_0, thresholds_0 = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    roc_auc_0 = auc(fpr_0, tpr_0)  # 得到area的数值

    plt.subplots(figsize=(7, 5.5))
    # 画出ROC曲线
    plt.plot(fpr_0, tpr_0, color='darkorange', lw=2, label='  area = %0.4f' % roc_auc_0)
    # 蓝色的线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # 图像标注
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('./fig_'+str(save_name), dpi=200)


def delong_test(labels, prediction_1, prediction_2):
    log10_p = delong_roc_test(labels, prediction_1, prediction_2)
    p = 10**log10_p  # 由于原函数中是log形式，所以使用指数运算将其转换
    print("p-value is", p)
    return p

# ===================It's all test code======================
# # test code
# prediction_1 = RF_grid_train()
# print(prediction_1)
# # TP, TN, FP, FN = conf_count(prediction_1, y_test_v)
# print_roc(prediction_1, y_test_v, 'model1')
#
# prediction_2 = SVM_grid_train()
# print(prediction_2)
# # TP, TN, FP, FN = conf_count(prediction_2, y_test_v)
# print_roc(prediction_2, y_test_v, 'model2')
# # help(plt.savefig)
# # plt.savefig('./test2.jpg')
#
# labels = y_test_v
#
# log10_p = delong_roc_test(labels, prediction_1, prediction_2)
# p = 10**log10_p  # 由于原函数中是log形式，所以使用指数运算将其转换
# print("p-value is", p)
# # 显著性水平=0.05， H0假设为两模型效果相同，但是P=0.48，所以拒绝原假设，认为模型的性能是不同的