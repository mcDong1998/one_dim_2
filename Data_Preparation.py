import os
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def read_data_file():
    """读取文件，返回读取到的数据"""
    data_filename = r'E:\Data\Projects\race\UI\Dmc_project\initial_stage_CKD\ES1.csv'
    if not os.path.exists(data_filename):
        print("Error: Missing file \'%s\'" % data_filename)
        return
    # pandas 读取csv文件：向量维度：1*1*8 ,列表用法
    test_data = pandas.read_csv(data_filename, names=["Flag", "L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"])
    return test_data


def data_prepare(random_num):
    """将读取到的csv文件，保留数值元素，去掉标签等非数值元素"""
    value_data = read_data_file()
    value_data = value_data.values
    print(value_data.shape)

    label = value_data[:, 0]  # 取第一列元素
    datav = value_data[:, 1:]  # 取第一列之外剩余的元素
    scaler = StandardScaler()
    scaler.fit(datav)
    datas = scaler.transform(datav)  # 为什么要用transform

    x_train, x_test, y_train, y_test = train_test_split(datas, label, test_size=0.2, random_state=random_num, stratify=label)
    x_verify = x_test.copy()
    y_verify = y_test.copy()

    return x_train, x_test, y_train, y_test, x_verify, y_verify


class OneDimData(Dataset):  # 继承Dataset

    def __init__(self, datav, label, transform=None):  # __init__是初始化该类的一些基础参数

        self.data = datav
        self.label = label
        self.transform = transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        sample = self.data[index, :]  # 根据索引index获取
        sampleflag = self.label[index]
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换

        return sample.astype(np.float32), sampleflag.astype(np.int64)  # 返回该样本


# test code
# x_train, x_test, y_train, y_test, x_verify, y_verify = data_prepare()
# print('x_train.shape', x_train.shape)
# print('x_test.shape', x_test.shape)
# print('y_train.shape', y_train.shape)
# print('y_test.shape', y_test.shape)
# print('x_verify.shape', x_verify.shape)
# print('y_verify.shape', y_verify.shape)
#
