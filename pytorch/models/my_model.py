import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # (16, 32, 32)
            nn.MaxPool2d(kernel_size=3, stride=2) # choose max value in 2x2 area, output shape (16, 15, 15)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # output shape (64, 15, 15)
            nn.ReLU()  # (16, 32, 32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # output shape (32, 15, 15)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # output shape (32, 15, 15)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # output shape (32, 15, 15)
            nn.MaxPool2d(kernel_size=3, stride=2)  # choose max value in 2x2 area, output shape (32, 7, 7)
        )
        self.fc1 = nn.Linear(64 * 7* 7, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.out = nn.Linear(1000, 10)  # fully connected layer, output 10 classes


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        outs = self.out(x)
        return outs
