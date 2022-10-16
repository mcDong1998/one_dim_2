from scipy.stats import shapiro
from scipy.stats import kstest
import scipy.stats as stats
import os
import pandas
from scipy.stats import chi2_contingency
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

def read_data_file(sheet_name):
    """读取文件，返回读取到的数据"""
    data_filename = r'E:\Pycharm_data\pythonProject2\test_final_2.xlsx'
    if not os.path.exists(data_filename):
        print("Error: Missing file \'%s\'" % data_filename)
        return
    # pandas 读取csv文件：向量维度：1*1*8 ,列表用法
    test_data = pandas.read_excel(data_filename, sheet_name=sheet_name)
    return test_data


def data_prepare(sheet_name):
    """将读取到的csv文件，保留数值元素，去掉标签等非数值元素
    sheet_name，用于选择特定的表格"""
    value_data = read_data_file(sheet_name)
    value_data = value_data.values
    print(value_data.shape)

    datav = value_data[:, 1:]  # 取第一列之外剩余的元素
    return datav


def normal_plot(data):
    """画出数据的分布，传入的参数是data，pd.dataframe形式
    例：
    score_df = pd.DataFrame()
    score_df['No_1'] = [0.946, 0.964, 0.893, 0.973, 0.938, 0.938, 0.938, 0.938, 0.938, 0.946]
    normal_plot(score_df['No_1'])
    """
    fg = plt.figure(figsize=(10, 5))
    sns.distplot(data)
    sns.set_palette('summer')
    plt.show()


# data = data_prepare('Se_DenseNet121')
# normal_plot(data[:, 8])

def ks_normal_detection(data):
    """使用KS检验去测试excel中的一列数据是否符合正则分布，样本量大于几十时可以使用——用不上
    例：
    data = pd.read_excel('a.xlsx')
    ks_normal_detection(data)
    """
    b = kstest(data.年龄, cdf="norm")
    print(b)
    c = kstest(data.收入, cdf="norm")
    print(c)
    return b, c


def w_normal_detection_example():
    """使用w检验去测试excel中的两列数据的差值是否符合正则分布, 传入的参数为excel形式的文件，样本量比较小时适用
    当然也可以只检测一组数据是否来自于正态分布"""
    data = data_prepare('ACC')
    data_3 = data[:, 2]
    data_5 = data[:, 4]
    p1 = shapiro(data_3-data_5)

    data = data_prepare('Recall')
    data_3 = data[:, 2]
    data_5 = data[:, 4]
    p2 = shapiro(data_3-data_5)

    data = data_prepare('Precision')
    data_3 = data[:, 2]
    data_5 = data[:, 4]
    p3 = shapiro(data_3-data_5)

    data = data_prepare('SPE')
    data_3 = data[:, 2]
    data_5 = data[:, 4]
    p4 = shapiro(data_3-data_5)

    data = data_prepare('F1')
    data_3 = data[:, 2]
    data_5 = data[:, 4]
    p5 = shapiro(data_3-data_5)

    data = data_prepare('AUC')
    data_3 = data[:, 2]
    data_5 = data[:, 4]
    p6 = shapiro(data_3-data_5)
    print(p1, p2, p3, p4, p5, p6)


# w_normal_detection_example()
# data = data_prepare('Se_DenseNet121')
# data_1 = data[:, 10]
# p1 = shapiro(data_1)
# print(p1)
# w_normal_detection()
# help(shapiro)

def K_W_test_example():
    """调用自己的数据进行检验：
    K-W检验是非参数检验方法，用于推断计量资料或等级资料的多个独立样本所来自的多个总体分布是否有差别
    本例中检验不同模型的多个指标：ACC，Recall，Pre等指标是否有差别——属于非参数检验方法
    eg: K_W_test_example()"""
    data = data_prepare('ACC')
    data_1 = data[:, 0]
    data_2 = data[:, 1]
    data_3 = data[:, 2]
    data_4 = data[:, 3]
    data_5 = data[:, 4]
    sta, p_value = stats.kruskal(data_3, data_5)
    print('ACC sta, p_value is ', sta, p_value)

    data = data_prepare('Recall')
    data_1 = data[:, 0]
    data_2 = data[:, 1]
    data_3 = data[:, 2]
    data_4 = data[:, 3]
    data_5 = data[:, 4]
    sta, p_value = stats.kruskal(data_3, data_5)
    print('Recall sta, p_value is ', sta, p_value)

    data = data_prepare('Precision')
    data_1 = data[:, 0]
    data_2 = data[:, 1]
    data_3 = data[:, 2]
    data_4 = data[:, 3]
    data_5 = data[:, 4]
    sta, p_value = stats.kruskal(data_3, data_5)
    print('Precision sta, p_value is ', sta, p_value)

    data = data_prepare('SPE')
    data_1 = data[:, 0]
    data_2 = data[:, 1]
    data_3 = data[:, 2]
    data_4 = data[:, 3]
    data_5 = data[:, 4]
    sta, p_value = stats.kruskal(data_3, data_5)
    print('SPE sta, p_value is ', sta, p_value)

    data = data_prepare('F1')
    data_1 = data[:, 0]
    data_2 = data[:, 1]
    data_3 = data[:, 2]
    data_4 = data[:, 3]
    data_5 = data[:, 4]
    sta, p_value = stats.kruskal(data_3, data_5)
    print(' F1 sta, p_value is ', sta, p_value)

    data = data_prepare('AUC')
    data_1 = data[:, 0]
    data_2 = data[:, 1]
    data_3 = data[:, 2]
    data_4 = data[:, 3]
    data_5 = data[:, 4]
    sta, p_value = stats.kruskal(data_3, data_5)
    print(' AUC sta, p_value is ', sta, p_value)


K_W_test_example()

def t_test_example():
    """为了说明数据是否有统计意义，需要先检验 数据是否符合正态分布：
    如果符合：使用小样本t检验，说明两组数据的均值有无差异——本例为样本符合正态分布的t-test检验
    如果不符合：使用W符号秩检验，说明两组数据的中位数有无差异
    当然也可以用来计算一组来自正态分布的样本的总体均值以及置信区间

    t-test：检验 KNCNN 与 RF 的 各项指标有无统计学差异"""
    # H0假设为两种方法相同，总之是所希望求得的假设的反面
    data = data_prepare('ACC')
    data_3 = data[:, 2]  # acc of KNCNN
    data_5 = data[:, 4]  # acc of RF
    data_sub = data_3-data_5
    # print(data_sub)
    t, p_two = ss.ttest_rel(data_3, data_5)  # ttest_rel:相关样本t检验
    t_chart = 2.262
    se = ss.sem(data_sub)  # 计算标准误差
    a = data_sub.mean() - t_chart * se  # 置信区间上限
    b = data_sub.mean() + t_chart * se  # 置信区间下限
    print('ACC p_two， CI is', p_two, a, b)

    data = data_prepare('Recall')
    data_3 = data[:, 2]  # acc of KNCNN
    data_5 = data[:, 4]  # acc of RF
    data_sub = data_3 - data_5
    # print(data_sub)
    t, p_two = ss.ttest_rel(data_3, data_5)  # ttest_rel:相关样本t检验
    t_chart = 2.262
    se = ss.sem(data_sub)  # 计算标准误差
    a = data_sub.mean() - t_chart * se  # 置信区间上限
    b = data_sub.mean() + t_chart * se  # 置信区间下限
    print('Recall p_two， CI is', p_two, a, b)

    data = data_prepare('Precision')
    data_3 = data[:, 2]  # acc of KNCNN
    data_5 = data[:, 4]  # acc of RF
    data_sub = data_3 - data_5
    # print(data_sub)
    t, p_two = ss.ttest_rel(data_3, data_5)  # ttest_rel:相关样本t检验
    t_chart = 2.262
    se = ss.sem(data_sub)  # 计算标准误差
    a = data_sub.mean() - t_chart * se  # 置信区间上限
    b = data_sub.mean() + t_chart * se  # 置信区间下限
    print('Precision p_two， CI is', p_two, a, b)

    data = data_prepare('SPE')
    data_3 = data[:, 2]  # acc of KNCNN
    data_5 = data[:, 4]  # acc of RF
    data_sub = data_3 - data_5
    # print(data_sub)
    t, p_two = ss.ttest_rel(data_3, data_5)  # ttest_rel:相关样本t检验
    t_chart = 2.262
    se = ss.sem(data_sub)  # 计算标准误差
    a = data_sub.mean() - t_chart * se  # 置信区间上限
    b = data_sub.mean() + t_chart * se  # 置信区间下限
    print('SPE p_two， CI is', p_two, a, b)

    data = data_prepare('F1')
    data_3 = data[:, 2]  # acc of KNCNN
    data_5 = data[:, 4]  # acc of RF
    data_sub = data_3 - data_5
    # print(data_sub)
    t, p_two = ss.ttest_rel(data_3, data_5)  # ttest_rel:相关样本t检验
    t_chart = 2.262
    se = ss.sem(data_sub)  # 计算标准误差
    a = data_sub.mean() - t_chart * se  # 置信区间上限
    b = data_sub.mean() + t_chart * se  # 置信区间下限
    print('F1 p_two， CI is', p_two, a, b)

    data = data_prepare('AUC')
    data_3 = data[:, 2]  # acc of KNCNN
    data_5 = data[:, 4]  # acc of RF
    data_sub = data_3 - data_5
    # print(data_sub)
    t, p_two = ss.ttest_rel(data_3, data_5)  # ttest_rel:相关样本t检验
    t_chart = 2.262
    se = ss.sem(data_sub)  # 计算标准误差
    a = data_sub.mean() - t_chart * se  # 置信区间上限
    b = data_sub.mean() + t_chart * se  # 置信区间下限
    print('AUC p_two， CI is', p_two, a, b)


def chi2_1(data):
    # 未校正的卡方  结果依次是:卡方值, p值, 自由度, 期望频率
    a, b, c, d = chi2_contingency(data, False)
    return a, b, c, d


def chi2_2(data):
    # 校正过的卡方  结果依次是:卡方值, p值, 自由度, 期望频率
    a, b, c, d = chi2_contingency(data, True)
    return a, b, c, d








