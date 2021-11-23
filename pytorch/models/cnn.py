import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.conv1 =nn.Sequential(         # input shape (3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),                    #(32, 26, 26)
            nn.MaxPool2d(2, 2),    # choose max value in 2x2 area, output shape (16, 13, 13)
        )
        self.out = nn.Linear(32*13*13, 10)
        nn.init.kaiming_normal_(self.out.weight)


    def forward(self, x):
        outs = None
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        outs = self.out(x)

        return outs
