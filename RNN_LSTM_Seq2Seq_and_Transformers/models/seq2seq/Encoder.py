import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint.                                      #
        #       4) A dropout layer                                                  #
        #############################################################################
        self.embedding = nn.Embedding(input_size, emb_size)
        if model_type=="RNN":
            self.rnn = nn.RNN(emb_size, encoder_hidden_size, batch_first=True)
        if model_type=="LSTM":
            self.rnn = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)
        self.l1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len, input_size)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        output, hidden = None, None
        embedded = self.embedding(input)
        embedded = self.drop(embedded)
        output, hidden = self.rnn(embedded)
        hidden = self.l1(hidden)
        hidden = self.act1(hidden)
        hidden = self.l2(hidden)
        # hidden = self.drop(hidden)
        act2 = nn.Tanh()
        hidden = act2(hidden)
        return output, hidden
