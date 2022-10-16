from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from KNCNN import *
from KD_DRSN import *
from KNCNN import *
from KD_IDA import *
from KD_IDA import *
from Data_Preparation import *

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 2.正则化：一定要注意transforms的用法，目前还不是特别清楚，但是一定要有两个以及以上的键值对，否则就会出问题。
# 当在进行enumerate的使用的时候，我只使用了一个transforms=...结果出现‘list is not callable'的错误。


# dataset = r'E:\Data\Projects\race\UI\Dmc_project\exp1_test_11_13\test'
#
# batch_size = 1
# num_classes = 2
# feature_extract = True
# data = {
#     'test': datasets.ImageFolder(root=dataset, transform=image_transforms['test']),
# }
# test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)


def conf_count(n):
    x_train, x_test, y_train, y_test, x_verify, y_verify = data_prepare(2021)
    xtest = OneDimData(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=xtest, batch_size=1, shuffle=True)

    path = r'E:\Pycharm_data\pythonProject2\KD_IDA_data\model_best'+str(n)+'.pt'
    model = torch.load(path)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # 0代表net，negative; 1代表spt，positive
    with torch.no_grad():
        for data in test_loader:

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # print(put)

            for j in range(1):
                if labels == 0 and outputs.argmax(dim=1)[j] == 0:
                    tp += 1
                elif labels == 0 and outputs.argmax(dim=1)[j] == 1:
                    fn += 1
                elif labels == 1 and outputs.argmax(dim=1)[j] == 0:
                    fp += 1
                elif labels == 1 and outputs.argmax(dim=1)[j] == 1:
                    tn += 1

    return tp, tn, fp, fn


TP = []
TN = []
FP = []
FN = []

for i in range(10):
    Tp, Tn, Fp, Fn = conf_count(i)
    TP.append(Tp)
    TN.append(Tn)
    FP.append(Fp)
    FN.append(Fn)

writer = pd.ExcelWriter(r'E:\Pycharm_data\pythonProject2\KD_IDA_data\KD_IDA_data.xlsx', index=None)
conf = np.zeros((10, 4))  # 由于索引是从0开始，所以当设置为11时，excel中的顺序会从0-10（这样的话会空出0行）
conf[:, 0] = TP
conf[:, 1] = TN
conf[:, 2] = FP
conf[:, 3] = FN
df = pd.DataFrame(conf)
df.to_excel(writer)
writer.save()

# def confusion_matrix():
#     path = 'E:\Data\code\Mr.ma\model_save\KNCNN\model_best.pt'
#
#     model = torch.load(path)
#     # print(type(model))
#     device = torch.device('cuda:0')
#     model = model.to(device)
#     model.eval()
#     y_true = []
#     y_score = []
#     with torch.no_grad():
#
#         for step, (b_x, b_y) in enumerate(test_):  # for j ,(in,label)
#             b_x = b_x.to(device)
#             b_y = b_y.to(device)
#             outputs = model(b_x)
#
#             for i in range(batch_size):
#                 y_true.append(b_y.item())
#                 y_score.append(F.softmax(outputs[i], dim=0)[1].item())

