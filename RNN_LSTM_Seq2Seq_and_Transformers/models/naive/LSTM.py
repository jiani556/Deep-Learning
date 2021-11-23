import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations

    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        ################################################################################
        #Input gate
        self.weight_ii = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.weight_hi = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_ii = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias_hi = nn.Parameter(torch.Tensor(self.hidden_size))

        #Forget Gate
        self.weight_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_if = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias_hf = nn.Parameter(torch.Tensor(self.hidden_size))

        #cell gate
        self.weight_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ig = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias_hg = nn.Parameter(torch.Tensor(self.hidden_size))

        #output gate
        self.weight_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_io = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias_ho = nn.Parameter(torch.Tensor(self.hidden_size))

        self.active_it = nn.Sigmoid()
        self.active_ft = nn.Sigmoid()
        self.active_gt = nn.Tanh()
        self.active_ot = nn.Sigmoid()
        self.active_ht = nn.Tanh()

        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        # 𝑖𝑡 = 𝜎(𝑊𝑖𝑖𝑥𝑡 + 𝑏𝑖𝑖 + 𝑊ℎ𝑖ℎ𝑡−1 + 𝑏ℎ𝑖)
        # 𝑓𝑡 = 𝜎( 𝑊𝑖𝑓𝑥𝑡 + 𝑏𝑖𝑓 + 𝑊ℎ𝑓ℎ𝑡−1 + 𝑏ℎ𝑓)
        # 𝑔𝑡 = tanh(𝑊𝑖𝑔𝑥𝑡 + 𝑏𝑖𝑔 + 𝑊ℎ𝑔ℎ𝑡−1 + 𝑏ℎ𝑔)
        # 𝑜𝑡 = 𝜎(𝑊𝑖𝑜𝑥𝑡 + 𝑏𝑖𝑜 + 𝑊ℎ𝑜ℎ𝑡−1 + 𝑏ℎ𝑜)
        # 𝑐𝑡 = 𝑓𝑡⊙𝑐𝑡−1 + 𝑖𝑡⊙𝑔𝑡
        # ℎ𝑡 = 𝑜𝑡⊙tanh(𝑐𝑡)
        h_t, c_t = torch.zeros(self.hidden_size), torch.zeros(self.hidden_size)
        batchSize, stepSize, featureSize = x.size()
        for step in range(stepSize):
            x_t = x[:, step, :]
            #input gate
            i_t = self.active_it(x_t @ self.weight_ii + self.bias_ii + h_t @ self.weight_hi + self.bias_hi)
            #forget gate:
            f_t = self.active_ft(x_t @ self.weight_if + self.bias_if + h_t @ self.weight_hf + self.bias_hf)
            #cell gate:
            g_t = self.active_gt(x_t @ self.weight_ig + self.bias_ig + h_t @ self.weight_hg + self.bias_hg)
            #output gate:
            o_t = self.active_ot(x_t @ self.weight_io + self.bias_io + h_t @ self.weight_ho + self.bias_ho)
            #c_t
            c_t = f_t * c_t + i_t * g_t  # LSTM is additive since it adds input info
            # external hidden state
            h_t = o_t * self.active_ht(c_t)

        return (h_t, c_t)
