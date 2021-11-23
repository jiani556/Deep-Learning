import random

import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """
    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout = 0.2, model_type = "RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #############################################################################

        self.embedding = nn.Embedding(output_size, emb_size)
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)
        if model_type == "LSTM":
            self.rnn = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)
        self.l1 = nn.Linear(decoder_hidden_size, output_size)
        self.act1 = nn.LogSoftmax(dim=-1)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        output = None
        batch_size =input.size()[0]
        embedded = self.embedding(input)
        embedded = self.drop(embedded)
        output, hidden = self.rnn(embedded,hidden)
        output = self.l1(output)
        output = self.act1(output)
        output = output.view(batch_size,self.output_size)
        return output, hidden
