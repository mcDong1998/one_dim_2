"""使用CNN对一维数据进行分类实验，包括：训练函数定义，混淆矩阵函数定义,如果要换网络模型，更改第27行即可。"""
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
# from KNCNN import *
from KD_DRSN import *
from KNCNN import *
from KD_IDA import *
# from KD_IDA import *
from Data_Preparation import *
# from Test import *
#
# # hyper parameters
# max_epoch = 100  # 训练整批数据多少次
# batch_size = 32
# lr_init = 0.001  # 学习率
#
#
# # data
# x_train, x_test, y_train, y_test, x_verify, y_verify = data_prepare()
# xtrain = OneDimData(x_train, y_train)
# xtest = OneDimData(x_test, y_test)
# train_loader = torch.utils.data.DataLoader(dataset=xtrain, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=xtest, batch_size=batch_size, shuffle=True)
#
#
# # neural network model
# cnn = DRSN()
#
# # loss and optimizer
# loss_function = nn.CrossEntropyLoss()  # 选择损失函数
# optimizer = optim.SGD(cnn.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略
#
# # record parameter
# now_time = datetime.now()
# time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
# writer = SummaryWriter(log_dir="./")


def train(i):
    history = []

    """定义训练函数，所需要的变量基本都在外部进行定义了"""
    for epoch in range(max_epoch):

        # count my own index
        train_loss = 0.0
        test_loss = 0.0

        loss_sigma = 0.0  # 记录一个epoch的loss之和
        correct = 0.0
        total = 0.0
        max_acc = 0
        # 训练模式
        for step, (b_x, b_y) in enumerate(train_loader):  # for j ,(in,label)
            # print('step', step)
            optimizer.zero_grad()
            outputs = cnn(b_x)
            loss = loss_function(outputs, b_y)
            loss.backward()
            optimizer.step()

            # add here
            train_loss += loss.item() * b_y.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += b_y.size(0)
            # print('b_y.size(0)', b_y.size(0))
            correct += (predicted == b_y).squeeze().sum().numpy()
            loss_sigma += loss.item()

            if step % 10 == 5:  # 如果到了第5个批量，即最后一批数据
                loss_avg = loss_sigma / 10  # 为什么是除以10？
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4%}".format(
                    epoch + 1, max_epoch, step + 1, len(train_loader), loss_avg, correct / total))
                print('train_loss_avg', loss_avg)
        scheduler.step()  # 更新学习率
        loss_sigma = 0.0

        # 评估模式
        cnn.eval()
        outputs = cnn(x_verify.astype(np.float32))
        outputs.detach()

        y2 = torch.from_numpy(y_verify.astype(np.int64))
        y2 = torch.reshape(y2, (-1, 1))
        y2 = y2.squeeze_()
        loss = loss_function(outputs, y2)
        # add code here
        test_loss += loss.item() * y2.size(0)

        loss_sigma += loss.item()

        cnn.train()

        test_out = outputs
        predict_y = torch.argmax(test_out, 1).data.numpy()

        current_acc = sum(predict_y==y_test) * 100 / len(predict_y)
        if current_acc > max_acc:
            max_acc = current_acc
            max_test = predict_y
            net_save_path = os.path.join(r'E:\Pycharm_data\pythonProject2\KNCNN_data', 'model_best'+str(i)+'.pt')
            torch.save(cnn, net_save_path)
        if epoch == (max_epoch-1):
            net_save_path = os.path.join(r'E:\Pycharm_data\pythonProject2\KNCNN_data', 'model_last'+str(i)+'.pt')
            torch.save(cnn, net_save_path)
        # history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
        print('Epoch={} -  Test set Accuracy:{:.4%}'.format(epoch, current_acc / 100.0))

        # add code here
        avg_train_loss = train_loss/162
        avg_test_loss = test_loss/41
        history.append([avg_train_loss, avg_test_loss])

    print('finished training')
    print('ACC= %.4f%%' % max_acc)

    return max_test, predict_y, history  # 返回值是最好的Y的预测以及最后一次Y的预测


def confusion_matrix(y_prediction, y_test):
    """ I use a trick to count the confusion matrix, but here is a question:
    函数中使用的数据y_prediction是上述train函数的返回值，从而是最好模型的预测值，
    但是这么做相当于导致测试集数据集泄露，因此需要先保存最后一次训练的模型（而不是测试集准确率最高的模型）
    然后再调用保存的最后一次训练的模型，进行混淆矩阵数据以及AUC数值的计算。
    *所以，train函数需要修改,maybe all the data need to be updated...I have a lot of work to do...
    I don't think I can finish them this month...(11.22)"""
    a = y_prediction - y_test
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(41):
        if a[j] == 1:
            fn += 1
        if a[j] == -1:
            fp += 1
        tp = 22 - fn
        tn = 19 - fp
    return tp, tn, fp, fn


def print_roc_1(y_pred_test, y_test, name):
    y_true = y_test  # 非二进制需要pos_label
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
    plt.savefig('./fig_'+str(name), dpi=200)


def print_roc_2():
    path = 'E:\Data\code\Mr.ma\model_save\KD_IDA\model_best.pt'

    model = torch.load(path)
    # print(type(model))
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():

        for step, (b_x, b_y) in enumerate(train_loader):  # for j ,(in,label)
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            outputs = model(b_x)

            for i in range(batch_size):
                y_true.append(b_y.item())
                y_score.append(F.softmax(outputs[i], dim=0)[1].item())

    fpr_0, tpr_0, thresholds_0 = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    roc_auc_0 = auc(fpr_0, tpr_0)

    plt.subplots(figsize=(7, 5.5));
    # 画出ROC曲线
    plt.plot(fpr_0, tpr_0, color='darkorange', lw=2, label=' KD_IDA (area = %0.4f)' % roc_auc_0)
    # 蓝色的线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # 图像标注
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
# y_prediction = train()
# confusion_matrix(y_prediction, y_test)
def plot_loss(history):
    """
   （1）dataset为输入图片要保存的路径名称，
   （2）history是列表形式的文件，保存了epochs*4(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc)的数据
       history中的数据由train函数返回。
   （3）函数无返回值
    (4)调用示例：plot_loss_and_acc(dataset,history)"""
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Train Loss', 'Validation loss'])
    plt.xlabel('epoch number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    # plt.savefig(dataset + '_loss_curve.png')
    plt.show()


for i in range(10):

    TP = []
    TN = []
    FP = []
    FN = []

    # hyper parameters
    max_epoch = 100  # 训练整批数据多少次
    batch_size = 32
    lr_init = 0.0001  # 学习率

    # data
    x_train, x_test, y_train, y_test, x_verify, y_verify = data_prepare(2021)
    xtrain = OneDimData(x_train, y_train)
    xtest = OneDimData(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=xtrain, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=xtest, batch_size=batch_size, shuffle=True)

    # neural network model
    cnn = KNCNN()

    # loss and optimizer
    loss_function = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(cnn.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

    max_test, predict_y, history = train(i)  # max_test 为模型最好一次在测试集的表现，predict_y为最后一次模型在测试集上的表现
    # plot_loss(history)
    tp, tn, fp, fn = confusion_matrix(predict_y, y_test)
    #
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)

    print_roc_1(predict_y, y_test, 'KNCNN' + str(i))  # 保存ROC图像 使用简单的方法，不必再重新加载模型


# writer = pd.ExcelWriter(r'E:\Pycharm_data\pythonProject2\KNCNN_data.xlsx', index=None)
# conf = np.zeros((10, 4))  # 由于索引是从0开始，所以当设置为11时，excel中的顺序会从0-10（这样的话会空出0行）
# conf[:, 0] = TP
# conf[:, 1] = TN
# conf[:, 2] = FP
# conf[:, 3] = FN
# df = pd.DataFrame(conf)
# df.to_excel(writer)
# writer.save()

    # # record parameter
    # now_time = datetime.now()
    # time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    # writer = SummaryWriter(log_dir="./")




