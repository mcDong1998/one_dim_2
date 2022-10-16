import torch
import torch.nn as nn
import torch.nn.functional as F


class Node_1(nn.Module):
    def __init__(self,
                cin: int = 3,
                cout : int = 3,
                ):
        super(Node_1, self).__init__()
        self.node = nn.Sequential(
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        x1 = self.node(x)
        x2 = F.relu(x1)
        # print(x2.shape)
        return x2


class Node_2(nn.Module):
    def __init__(self,
                 cin: int = 3,
                 cout: int = 3,
                 ):
        super(Node_2, self).__init__()
        self.node = nn.Sequential(
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        x1 = self.node(x)
        x2 = F.relu(x1)
        return x2

class Node_3(nn.Module):
    def __init__(self,
                     cin: int = 3,
                     cout: int = 3,
                 ):
        super(Node_3, self).__init__()
        self.node = nn.Sequential(
                nn.Conv2d(cin, cout, 1),
                nn.BatchNorm2d(cout, eps=1e-5, momentum=0.01, affine=True)
            )

    def forward(self, x):
        x1 = self.node(x)
        x2 = F.relu(x1)
        return x2


class KDblock(nn.Module):
    def __init__(
            self,
            cin: int = 3,
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
        self.node = Node_1(3, 3)

        self.gap = nn.AdaptiveAvgPool2d(1)  # output size=1*1
        # self.dropout = nn.Dropout(0.5)
        # 156的数字由计算得出
        self.linear1 = nn.Linear(3132, 100)
        self.dropout = nn.Dropout(0.5)
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

        x4_in = torch.cat((x1, x2), 3)
        x4_out = self.node(x4_in)

        x3_in = torch.cat((x4_out, x2), 3)
        x3_out = self.block3(x3_in)

        x5_in = torch.cat((x3_out, x4_out), 3)
        x5_out = self.node(x5_in)

        x6_in = torch.cat((x4_out, x5_out),3)
        x6 = self.node(x6_in)

        x = x6.view(x6.shape[0], x6.shape[3], x6.shape[1], -1)
        # print('x.shape', x.shape)

        x = self.gap(x)

        x = x.view(x.shape[0], x.shape[2], x.shape[3], -1)
        # print('x.shape', x.shape)

        x = self.linear1(x)
        # print('x.shape', x.shape)
        x = self.dropout(x)

        x = self.linear2(x)
        # print('x.shape', x.shape)

        x = torch.reshape(x, (batch, -1))
        # print('x.shape', x.shape)

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
# x = torch.rand(32, 3, 1, 240)
# net = Node_1()
# y = net(x)
# print(y.shape)

# x = torch.rand(32, 3, 1, 444)
# net = KDblock()
# y = net(x)
# print(y.shape)

# x = torch.rand(32, 1, 1, 8)
# net = KDnet()
# y = net(x)
# print(y.shape)