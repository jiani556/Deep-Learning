import random

import torch
import torch.nn as nn
import torch.optim as optim

# import custom models



class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)


    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]


        #############################################################################
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hideen state of the decoder                            #
        #       2) Starting from <sos> tokens, feed the tokens into the decoder     #
        #          a step at a time, add the output to the final outputs. Update    #
        #          the hidden weights and feeding tokens.                           #
        #############################################################################

        outputs = None
        SOS = source[0][0].item()
        encoder_output, encoder_hidden = self.encoder(source)
        decoder_hidden = encoder_hidden[-1,-1,].unsqueeze(0).unsqueeze(0)
        decoder_input = torch.tensor([[SOS]])
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        outputs = torch.zeros(out_seq_len, batch_size, decoder_output.size()[1])
        outputs[1] = decoder_output
        if out_seq_len>1:
            for i in range(2,out_seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # decoder_hidden = decoder_hidden[-1,].unsqueeze(0)
                outputs[i] = decoder_output
        outputs=outputs.permute(1,0,2)
        return outputs
