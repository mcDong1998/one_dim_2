"""修改之后，KD-IDA跑通了，Pytorch中变量有些奇怪，和代码的执行流程相关
    调用示例：
    x = torch.rand(32, 1, 1, 8)
    net = KDnet()
    y = net(x)
    print(y.shape)
"""
import torch
import torch.nn as nn


# n是输入向量的维度
class KDblock(nn.Module):
    def __init__(
            self,
            cin: int = 1,
            cout: int = 3,
    ):
        super(KDblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(cin, cout, 1),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 2)),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 3)),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 4)),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 2), dilation=2),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 3), dilation=2),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(input)
        x3 = self.conv3(input)
        x4 = self.conv4(input)
        x5 = self.conv5(input)
        x6 = self.conv6(input)
        x = torch.cat((x1, x2, x3, x4, x5, x6), 3)
        return x


class KDnet(nn.Module):

    def __init__(
            self,
            block1in: int = 1,
            block1out: int = 3,
            block2in: int = 3,
            block2out: int = 3,
            block3in: int = 3,
            block3out: int = 3,

    ):
        super(KDnet, self).__init__()

        self.block1 = KDblock(block1in, block1out)
        self.block2 = KDblock(block2in, block2out)
        self.block3 = KDblock(block3in, block3out)

        self.gap = nn.AdaptiveAvgPool2d(1)  # output size=1*1
        self.dropout = nn.Dropout(0.8)
        # 156的数字由计算得出
        self.linear1 = nn.Linear(1692, 100)
        self.linear2 = nn.Linear(100, 2)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        batch = x.shape[0]
        x = torch.reshape(x, (batch, 1, 1, -1))

        x1 = self.block1(x)
        # print('x1.shape', x1.shape)

        x2 = self.block2(x1)
        # print('x2.shape', x2.shape)

        x3 = self.block3(x2)
        # print('x3.shape', x3.shape)

        x4 = torch.cat((x1, x2), 3)
        # print('x4.shape', x4.shape)

        x5 = torch.cat((x3, x4), 3)
        # print('x5.shape', x5.shape)

        x6 = torch.cat((x4, x5), 3)
        # print('x6.shape', x6.shape)

        x = x6.view(x6.shape[0], x6.shape[3], x6.shape[1], -1)
        # print('x.shape', x.shape)

        x = self.gap(x)

        x = x.view(x.shape[0], x.shape[2], x.shape[3], -1)
        # print('x.shape', x.shape)

        x = self.linear1(x)
        # print('x.shape', x.shape)

        x = self.linear2(x)
        # print('x.shape', x.shape)

        x = torch.reshape(x, (batch, -1))
        # print('x.shape', x.shape)

        return x


# x = torch.rand(32, 1, 1, 8)
# net = KDnet()
# y = net(x)
# print(y.shape)


