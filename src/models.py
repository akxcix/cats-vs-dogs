import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 x 200 x 200
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 7 x 98 x 98

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # 2 x 47 x 47

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=4)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # 48 x 22 x 22
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        # 1 x 22 x 22

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        # 80 x 4 x 4

        self.dense6 = nn.Linear(80 * 4 * 4, 1)
        # 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dense6(x)
        x = torch.sigmoid(x)

        return x


class VGG1(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 x 200 x 200
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 32 x 99 x 99

        self.dense2 = nn.Linear(32 * 99 * 99, 128)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(128, 1)
        # 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = torch.sigmoid(x)

        return x
