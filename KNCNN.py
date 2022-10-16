"""本文件中定义了两个类，KNetBlock(), KNCNN() 测试方法在最后
调用示例：
    x = torch.rand(32, 1, 1, 8)
    net = KNCNN()
    y = net(x)
    print(y.shape)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KNetBlock(nn.Module):
    """KNetBlock 是KNCNN网络的基础模块，本函数将被调用3次"""
    def __init__(self, out_channel=1, in_shape = 8, out_shape=2,  display_shape_flag=False):
        super(KNetBlock, self).__init__()
        self.model_name = str(type(self))
        self.display_shape_flag = display_shape_flag
        self.out_shape = out_shape
        self.out_channel = out_channel
        self.in_shape = in_shape

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 2))
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 3))
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 4))
        self.bn4 = nn.BatchNorm2d(self.out_channel)

        self.conv2n = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 2), dilation=(1, 2))
        self.conv3n = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 3), dilation=(1, 2))
        self.conv4n = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 4), dilation=(1, 2))

        self.conv2n_maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv3n_maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv4n_maxpool = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2nn = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 2), dilation=(1, 3))
        self.conv3nn = nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(1, 3), dilation=(1, 3))
        self.conv2nn_maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv3nn_maxpool = nn.MaxPool2d(kernel_size=(1, 2))

    def forward(self, x):
        dis_shape = self.display_shape_flag
#        dis_shape = False
        batch = x.shape[0]
        if dis_shape:
            print(x.shape)
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
            if dis_shape:
                print(x.shape)

        x = torch.reshape(x, (batch, 1, 1, -1))
        if dis_shape:
            print(x.shape)
        # Convolution
        x1 = F.relu(self.bn1(self.conv1(x)))
        if dis_shape:
            print('x1 shape')
            print(x1.shape)
        x2 = F.relu(self.bn2(self.conv2(x)))
        if dis_shape:
            print('x2 shape')
            print(x2.shape)
        x3 = F.relu(self.bn3(self.conv3(x)))
        if dis_shape:
            print('x3 shape')
            print(x3.shape)
        x4 = F.relu(self.bn4(self.conv4(x)))
        if dis_shape:
            print('x4 shape')
            print(x4.shape)

        x2n = F.relu(self.conv2n(x))
        if dis_shape:
            print('x2n shape')
            print(x2n.shape)
        x3n = F.relu(self.conv3n(x))
        if dis_shape:
            print('x3n shape')
            print(x3n.shape)
        x4n = F.relu(self.conv4n(x))
        if dis_shape:
            print('x4n shape')
            print(x4n.shape)

        x2nn = F.relu(self.conv2nn(x))
        if dis_shape:
            print('x2nn shape')
            print(x2nn.shape)
        x3nn = F.relu(self.conv3nn(x))
        if dis_shape:
            print('x3nn shape')
            print(x3nn.shape)

        x2n_maxpool = F.relu(self.conv2n_maxpool(self.conv2n(x)))
        x3n_maxpool = F.relu(self.conv3n_maxpool(self.conv3n(x)))
        x4n_maxpool = F.relu(self.conv4n_maxpool(self.conv4n(x)))

        x2nn_maxpool = F.relu(self.conv2nn_maxpool(self.conv2nn(x)))
        x3nn_maxpool = F.relu(self.conv3nn_maxpool(self.conv3nn(x)))

        # capture and concatenate the features
        x1 = torch.reshape(x1, (batch, -1))
        x2 = torch.reshape(x2, (batch, -1))
        x3 = torch.reshape(x3, (batch, -1))
        x4 = torch.reshape(x4, (batch, -1))
        x2n = torch.reshape(x2n, (batch, -1))
        x3n = torch.reshape(x3n, (batch, -1))
        x4n = torch.reshape(x4n, (batch, -1))
        x2nn = torch.reshape(x2nn, (batch, -1))
        x3nn = torch.reshape(x3nn, (batch, -1))

        x2n_maxpool = torch.reshape(x2n_maxpool, (batch, -1))
        x3n_maxpool = torch.reshape(x3n_maxpool, (batch, -1))
        x4n_maxpool = torch.reshape(x4n_maxpool, (batch, -1))
        x2nn_maxpool = torch.reshape(x2nn_maxpool, (batch, -1))
        x3nn_maxpool = torch.reshape(x3nn_maxpool, (batch, -1))

        if dis_shape:
            print('x1 x2n x2nn x3nn shape')
            print(x1.shape)
            print(x2n.shape)
            print(x2nn.shape)
            print(x3nn.shape)
        x = torch.cat((x1, x2, x3, x4, x2n, x3n, x4n, x2nn, x3nn, x2n_maxpool, x3n_maxpool, x4n_maxpool,x2nn_maxpool, x3nn_maxpool),1)
#        print (x.shape)
        x = torch.reshape(x, (batch, -1))
#        print (x.shape)

        # project the features to the labels

        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class KNCNN(nn.Module):
    """KNCNN 由3个KNetBlock块和一层丢弃层和一层全连接层组成"""
    def __init__(self, out_channel=4, in_shape=8, out_shape=2, display_shape_flag=False):
        super(KNCNN, self).__init__()
        self.model_name = str(type(self))
        self.display_shape_flag = display_shape_flag
        self.out_shape = out_shape

        self.netblock1 = KNetBlock(out_channel=3, in_shape=8, out_shape=8, display_shape_flag=False)
        self.netblock2 = KNetBlock(out_channel=2, in_shape=8, out_shape=8, display_shape_flag=False)
        self.netblock3 = KNetBlock(out_channel=1, in_shape=8, out_shape=2, display_shape_flag=False)
        #
        # self.linear1 = nn.Linear(41937, 500)
        self.dropout1 = nn.Dropout(p=0.9)
        self.linear1 = nn.Linear(41937, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.netblock1(x)
        x = self.netblock2(x)
        x = self.netblock3(x)

        x = self.dropout1(x)
        x = self.linear1(x)

        # x = self.linear2(x)
        x = self.softmax(x)

        if self.display_shape_flag:
            print(x.shape)

        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


