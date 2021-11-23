# Code by Sarah Wiegreffe (saw@gatech.edu)
# Fall 2019

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        '''
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        seed_torch(0)

        self.word_embed = nn.Embedding(self.input_size, self.hidden_dim)
        self.position_embed = nn.Embedding(self.max_length, self.hidden_dim)

        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        self.l1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''

        outputs = None
        embeddings = self.embed(inputs)
        mha = self.multi_head_attention(embeddings)
        ff = self.feedforward_layer(mha)
        outputs = self.final_layer(ff)

        return outputs


    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """

        N, T = inputs.shape
        positions = torch.LongTensor(np.repeat([np.arange(0,T)],N,axis=0))
        embeddings = self.word_embed(inputs) + self.position_embed(positions)
        return embeddings

    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)

        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """

        outputs = None
        K1 = self.k1(inputs)
        Q1 = self.q1(inputs)
        V1 = self.v1(inputs)
        attention1 = torch.bmm(self.softmax(torch.bmm(Q1, K1.permute(0, 2, 1)) / np.sqrt(self.dim_k)), V1)
        K2 = self.k2(inputs)
        Q2 = self.q2(inputs)
        V2 = self.v2(inputs)
        attention2 = torch.bmm(self.softmax(torch.bmm(Q2, K2.permute(0, 2, 1)) / np.sqrt(self.dim_k)), V2)
        multi_head_attention = torch.cat((attention1, attention2), dim=2)
        outputs = self.attention_head_projection(multi_head_attention)
        outputs = self.norm_mh(outputs + inputs)
        return outputs


    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        outputs = None
        outputs = self.l1(inputs)
        outputs = self.act1(outputs)
        outputs = self.l2(outputs)
        outputs = self.norm_mh(outputs + inputs)

        return outputs


    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        outputs = None
        outputs = self.l3(inputs)
        return outputs


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
