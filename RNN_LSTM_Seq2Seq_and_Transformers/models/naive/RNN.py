import numpy as np
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """ An implementation of vanilla RNN using Pytorch Linear layers and activations.
        You will need to complete the class init function, forward function and hidden layer initialization.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
                output_size (int): the size of the output layer

            Returns:
                None
        """
        super(VanillaRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.ih = nn.Linear(input_size + hidden_size, hidden_size)
        self.ho = nn.Linear(input_size + hidden_size, output_size)
        self.act_hidden = nn.Tanh()
        self.act_output = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """ The forward function of the Vanilla RNN
            Args:
                input (tensor): a batch of data of shape (batch_size, input_size) at one time step
                hidden (tensor): the hidden value of previous time step of shape (batch_size, hidden_size)

            Returns:
                output (tensor): the output tensor of shape (batch_size, output_size)
                hidden (tensor): the hidden value of current time step of shape (batch_size, hidden_size)
        """

        output = None

        cat = torch.cat((input, hidden),dim=-1)
        with torch.no_grad():
            h = self.ih(cat)
            op = self.ho(cat)
        hidden=self.act_hidden(h)
        output=self.act_output(op)
    
        return output, hidden
