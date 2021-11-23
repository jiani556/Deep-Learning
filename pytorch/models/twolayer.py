import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.ac1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        out = None
        N = x.shape[0]
        x_reshaped= x.reshape(N, -1)
        out1=self.fc1(x_reshaped)
        act1=self.ac1(out1)
        out =self.fc2(act1)
        return out
