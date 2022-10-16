"""KD-DRSN网络模型的设计 :shrinkage被DRSNBlock调用，DRSNBlock被DRSN调用。
调用示例：
    net=DRSN()
    data = torch.randn(32, 1, 1, 8)
    preds = net(data)
    print(preds.shape)"""
import torch
import torch.nn as nn


class Shrinkage(nn.Module):
    """实现:C*W*1的从池化到软阈值的实现过程"""
    def __init__(self, gap_size=1, channel=1):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)  # output size=gap_size*gap_size
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, 1),  # 可能应该是nn.Linear(channel, channel)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x  # x_raw=1*w*1
        x = torch.abs(x)  # 取绝对值
        x_abs = x  # x_abs 副本=1*w*1
        x = self.gap(x)  # 池化 output_size=1*1*1
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)  # 逐元素相乘
        x = x.unsqueeze(2).unsqueeze(2)
        # 软阈值化
        sub = x_abs - x  # x就是keras代码中的阈值thres
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


# 需要的参数 cin=1,cout=1，已经设定好，不再需要其他参数
class DRSNBlock(nn.Module):
    def __init__(self, cin=1, cout=1):
        super(DRSNBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(cin, cout, 1),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(cin, cout, 1),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True))

        self.shrink = Shrinkage()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.shrink(x)
        x = identity + x
        x = self.relu(x)
        return x


# 需要的参数 cin=1,cout=1，已经设定好，不再需要其他参数
class DRSN(nn.Module):
    def __init__(self, cin=1, cout=1):
        super(DRSN, self).__init__()
        self.block = DRSNBlock()
        self.dropout = nn.Dropout(0.5)
        # 数字由计算得出
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 2)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        batch = x.shape[0]
        x = torch.reshape(x, (batch, 1, 1, -1))
        x = self.block(x)
        x = self.block(x)
        x = self.block(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.reshape(x, (batch, -1))
        return x

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





