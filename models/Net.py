import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 128)
        self.conv9 = DoubleConv(128,64)
        self.conv10 = nn.Conv2d(64,out_ch,1)
        # self.sigmoid = nn.Sigmoid()
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(192*64,192*64),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        c1 = self.conv1(x)
        # print(c1.shape)  # torch.Size([5, 64, 701, 255])
        p1 = self.pool1(c1)
        # print(p1.shape)  # torch.Size([5, 64, 350, 127])
        c2 = self.conv2(p1)
        # print(c2.shape)  # torch.Size([5, 128, 350, 127])
        p2 = self.pool2(c2)
        # print(p2.shape)  # torch.Size([5, 128, 175, 63])
        c3 = self.conv3(p2)
        # print(c2.shape)  # torch.Size([5, 128, 350, 127])
        p3 = self.pool3(c3)
        # print(p2.shape)  # torch.Size([5, 128, 175, 63])
        c4 = self.conv4(p3)
        # print(c4.shape)  # torch.Size([5, 512, 87, 31])
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        # print(up_6.shape)  # torch.Size([5, 512, 86, 30])
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        # out = c9
        c10 = self.conv10(c9)
        # c10 = self.sigmoid(c10)
        # print(c10.shape)
        # i=self.fc(c10)
        # i = i.reshape(16, 3)
        # print(i)
        # out = c10
        # out = nn.Softmax()(c10)
        return c10
if __name__ == '__main__':
    net=UNet(3,1)
    net=net.to("cuda")
    print(net)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x=np.random.randint(1,10,(3,16,96))#随机生成(3,10,401)的数组，数组中值的取值范围在【1，10】之间
    x=x.astype(np.float32)
    x = x[np.newaxis, :]
    x = torch.tensor(x)
    x=x.to("cuda")
    # print(x.shape)
    pred=net(x)
    print(pred.shape)