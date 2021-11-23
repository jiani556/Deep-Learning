
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import math

class PositionalEncoding(nn.Module):
    """
    reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    with modifications
    """
    def __init__(self, num_joints, hidden_dim, max_len=200):
        super(PositionalEncoding, self).__init__()
        d_model = num_joints*hidden_dim
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        pe = pe.reshape(pe.size(0), pe.size(1), num_joints, hidden_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        output = self.pe[:,:x.size(1), :]
        return output

class SpatialAttnLayer(nn.Module):
    def __init__(
            self,
            num_joints=24,
            num_heads=8,
            hidden_dim=128,
            dropout=0.1,
            use_torchMHA=False
    ):
        super(SpatialAttnLayer, self).__init__()

        self.num_joints = num_joints
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_torchMHA = use_torchMHA

        head_size = hidden_dim // num_heads
        self.q_dim = head_size
        self.k_dim = head_size
        self.v_dim = head_size

        self.q_mat = nn.Parameter(torch.zeros(num_heads, num_joints, hidden_dim, head_size))
        self.k_mat = nn.Parameter(torch.zeros(num_heads, hidden_dim, head_size))
        self.v_mat = nn.Parameter(torch.zeros(num_heads, hidden_dim, head_size))

        self.dropout = nn.Dropout(p=dropout)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.normlayer = nn.LayerNorm(hidden_dim)

    def multi_head_attention(self, input):
        batch_size, seq_len, _, _ = input.shape #(B,T,N,D)

        # q (B,T,H,N,S,1) =  q_mat_T (H,N,S,D) * input (B,T,1,N,D,1)
        q = torch.matmul(self.q_mat.transpose(2,3), torch.unsqueeze(torch.unsqueeze(input, -1), 2))
        q = torch.squeeze(q, -1)         # q (B,T,H,N,S)

        # K (B,T,H,N,S) = input (B,T,1,N,D) * k_mat (H,D,S)
        k = torch.matmul(input.unsqueeze(2), self.k_mat)

        # V (B,T,H,N,S) = input (B,T,1,N,D) * v_mat (H,D,S)
        v = torch.matmul(input.unsqueeze(2), self.v_mat)
    
        # attention (B,T,H,N,N) = q (B,T,H,N,S) * k_t (B,T,H,S,N)
        attention = torch.softmax(
        torch.div(torch.matmul(q, k.transpose(3, 4)), math.sqrt(self.k_dim)), -1)

        # x (B,T,H,N,S) = attention (B,T,H,N,N) * v (B,T,H,N,S)
        x = torch.matmul(attention, v)
        x = x.transpose(2,3)   # (B,T,N,H,S)

        output = x.reshape(batch_size, seq_len, self.num_joints, -1)
        output = self.attn_proj(output)
        return output

    def forward(self, input):
        input_src = input
        output = self.normlayer(self.dropout(self.multi_head_attention(input_src)) + input_src)
        return output

class TemporalAttnLayer(nn.Module):
    def __init__(
        self,
        num_joints=24,
        num_heads=8,
        hidden_dim=128,
        dropout=0.1,
        use_torchMHA=False
    ):
        """
        num_joints: number of joints, 24 for AMASS
        num_heads: number of attention heads
        hidden_dim: hidden embedding size
        dropout: dropout probablity
        use_torchMHA: whether use nn.MultiheadAttention or our implementation
        """
        super(TemporalAttnLayer, self).__init__()
        
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_torchMHA = use_torchMHA
        
        # initialize Q, K, V weight matrices
        # per joint, per head, weight matrix (hidden_dim, hidden_dim//num_heads)
        head_size = hidden_dim//num_heads
        self.q_dim = head_size
        self.k_dim = head_size
        self.v_dim = head_size
        
        if use_torchMHA:
            self.attn_layers = nn.ModuleList(
                [ nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    bias=False,
                    kdim=hidden_dim,
                    vdim=hidden_dim
                    )
                 for i in range(num_joints)
                    ]
                )
        else:
            self.q_mat = nn.Parameter(torch.zeros(num_heads, num_joints, hidden_dim, head_size)) # q_mat initiated to [H, N, D, F]
            self.k_mat = nn.Parameter(torch.zeros(num_heads, num_joints, hidden_dim, head_size)) # k_mat initiated to [H, N, D, F]
            self.v_mat = nn.Parameter(torch.zeros(num_heads, num_joints, hidden_dim, head_size)) # v_mat initiated to [H, N, D, F]
            self.attn_proj = nn.Parameter(torch.zeros(num_joints, hidden_dim, hidden_dim)) # attn_proj initiated to [N, D, D]
        
        self.dropout = nn.Dropout(p=dropout)
        self.normlayer = nn.LayerNorm(hidden_dim)
    
    def multi_head_attention(self, input, mask):
        """
        input: (batch_size, seq_len, num_joints, hidden_dim)
        mask: attention mask to avoid future info leakage
        
        output: same size as input, (batch_size, seq_len, num_joints, hidden_dim)
        """
        if self.use_torchMHA:
            input = input.transpose(0, 1)
            output = torch.zeros(input.shape).to(input.device)
            for idx, layer in enumerate(self.attn_layers):
                attn_input = input[:,:,idx,:]
                attn_output, _ = layer(attn_input, attn_input, attn_input, attn_mask=mask, need_weights=False)
                output[:,:,idx,:] = attn_output
            return output.transpose(0, 1)
        else:
            orig_shape = input.shape
            input = input.transpose(1, 2) # input shape changed from [B, T, N, D] -> [B, N, T, D]
            input = torch.unsqueeze(input, 1) # add one dimension to input to [B, 1, N, T, D]
            q = torch.matmul(input, self.q_mat) # matrix multiplication needed on [T * D] * [D * F] as in formula (3) of the paper, [B, 1, N, T, D] * [H, N, D, F] -> [B, H, N, T, F]
            k = torch.matmul(input, self.k_mat) # matrix multiplication needed on [T * D] * [D * F] as in formula (3) of the paper, [B, 1, N, T, D] * [H, N, D, F] -> [B, H, N, T, F]
            v = torch.matmul(input, self.v_mat) # matrix multiplication needed on [T * D] * [D * F] as in formula (3) of the paper, [B, 1, N, T, D] * [H, N, D, F] -> [B, H, N, T, F]
            attention = torch.softmax(torch.div(torch.matmul(q, k.transpose(3, 4)), math.sqrt(self.k_dim)) + mask, -1) # [B, H, N, T, F] * [B, H, N, F, T] -> attention of size [B, H, N, T, T]
            output = torch.matmul(attention, v) # [B, H, N, T, T] * [B, H, N, T, F] -> [B, H, N, T, F]
            output = output.transpose(1, 3) # [B, H, N, T, F] ->[B, T, N, H, F] so that we can concatenate all the heads 
            output = output.reshape(orig_shape) # [B, T, N, H, F] -> [B, T, N, D] 
            output = output.unsqueeze(3) # [B, T, N, D] -> [B, T, N, 1, D] 
            output = torch.matmul(output, self.attn_proj) # [B, T, N, 1, D] * [N, D, D] -> [B, T, N, 1, D] 
            output = output.reshape(orig_shape) # [B, T, N, 1, D] ->[B, T, N, D] 
            return output
    
        
    def forward(self, input):
        """
        input: from embedding/prev layer, tuple of (input_src, mask)
        
        output: same size as input_src, (batch_size, seq_len, num_joints, hidden_dim)
        """
        input_src, mask = input
        output = self.normlayer(self.dropout(self.multi_head_attention(input_src, mask)) + input_src)
        return output

class STAttnLayer(nn.Module):
    def __init__(
        self,
        num_joints=24,
        num_heads_t=8,
        num_heads_s=8,
        hidden_dim=128,
        feedfwd_dim=256,
        dropout=0.1,
        use_torchMHA=False
    ):
        """
        num_joints: number of joints, 24 for AMASS
        num_heads_t: number of temporal attention heads
        num_heads_s: number of spatial attention heads
        hidden_dim: hidden embedding size
        feedfwd_dim: feed forward layer after attention
        dropout: dropout probablity
        use_torchMHA: whether use nn.MultiheadAttention or our implementation
        """
        super(STAttnLayer, self).__init__()
        
        self.num_joints = num_joints
        self.num_heads_t = num_heads_t
        self.num_heads_s = num_heads_s
        self.hidden_dim = hidden_dim
        self.feedfwd_dim = feedfwd_dim
        self.use_torchMHA = use_torchMHA
        
        self.attnlayer_t = TemporalAttnLayer(
            num_joints,
            num_heads_t,
            hidden_dim,
            dropout,
            use_torchMHA
        )
        
        self.attnlayer_s = SpatialAttnLayer(
            num_joints,
            num_heads_s,
            hidden_dim,
            dropout,
            use_torchMHA
        )
        
        self.ff1 = nn.Linear(self.hidden_dim, self.feedfwd_dim)
        self.ff2 = nn.Linear(self.feedfwd_dim, self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.normlayer = nn.LayerNorm(hidden_dim)
    
    def forward(self, input):
        """
        input: from embedding/prev layer, tuple of (input_src, mask)

        output: tuple of forward output and mask
        """
        input_src, mask = input
        if self.attnlayer_s is not None and self.attnlayer_t is not None:
            attnlayer_t_output = self.attnlayer_t(input)
            attnlayer_s_output = self.attnlayer_s(input_src)
            attnlayer_output = attnlayer_t_output + attnlayer_s_output
        elif self.attnlayer_s is None:
            attnlayer_t_output = self.attnlayer_t(input)
            attnlayer_output = attnlayer_t_output
        elif self.attnlayer_t is None:
            attnlayer_s_output = self.attnlayer_s(input_src)
            attnlayer_output = attnlayer_s_output
        else:
            raise NotImplementedError
        
        ff_output = self.ff2(self.relu(self.ff1(attnlayer_output)))
        output = self.normlayer(attnlayer_output + self.dropout(ff_output))
        return output, mask
        
class STTransformerModel(nn.Module):
    def __init__(
        self,
        num_joints=24,
        rep_size=9,
        num_heads_t=4,
        num_heads_s=4,
        hidden_dim=64,
        feedfwd_dim=256,
        num_layers=4,
        dropout=0.1,
        use_torchMHA=False
    ):
        """
        Reference to Figure 2 (https://arxiv.org/abs/2004.08692)
        initialize following in order:
            Embedding layer:
                Joint Embedding
                Temporal Positional Encoding
            Attention layer:
                Temporal Attention
                Spatial Attention
            Output layer
        """
        super(STTransformerModel, self).__init__()
        
        self.num_joints = num_joints
        self.rep_size = rep_size
        self.num_heads_t = num_heads_t
        self.num_heads_s = num_heads_s
        self.hidden_dim = hidden_dim
        self.feedfwd_dim = feedfwd_dim
        self.num_layers = num_layers
        self.use_torchMHA = use_torchMHA
        
        self.joint_embedding = nn.Parameter(torch.zeros(num_joints, hidden_dim, rep_size))
        self.pos_encoder_t = PositionalEncoding(num_joints, hidden_dim)
        
        attnlayers = []
        for i in range(num_layers):
            attnlayers.append(
                STAttnLayer(
                    num_joints,
                    num_heads_t,
                    num_heads_s,
                    hidden_dim,
                    feedfwd_dim,
                    dropout,
                    use_torchMHA
                )
            )
        
        self.attnlayer = nn.Sequential(*attnlayers)
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(hidden_dim, rep_size)
        self.init_weights()
        
    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz):
        """
        mask with upper triangle shape to avoid future info leakage
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def run_forward(self, input, mask):
        """
        input: (B, T, E), E = num_joints * rep_size
        mask: attention mask
        
        output: (B, T, E)
        
        single forward run through all layers
        """
        input_v = input.reshape(input.shape[0], input.shape[1], self.num_joints, self.rep_size)
            
        joint_embedding = torch.matmul(self.joint_embedding, input_v.unsqueeze(-1)).squeeze(-1)
        input_embedded = self.dropout(joint_embedding + self.pos_encoder_t(joint_embedding))
            
        attn_output, _ = self.attnlayer((input_embedded, mask))
        output_v = input_v + self.projection(attn_output)
        return output_v.reshape(input.shape)
    
    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        """
        src: (B, T_src, E)
        tgt: (B, T_tgt, E)
        max_len: max predict len
        teacher_forcing_ratio: not used

        """
        # we use both src and tgt during training
        # i.e. we use input from 0 to (T_src + T_tgt - 2)
        # to predict 1 to (T_src + T_tgt - 1)
        if self.training:
            input = torch.cat((src, tgt), axis=1)[:,:-1]
            T = input.shape[1]
            mask = self._generate_square_subsequent_mask(T).to(src.device)
            output = self.run_forward(input, mask)
        else:
            if max_len is None:
                max_len = tgt.shape[1]
                
            T = src.shape[1]
            mask = self._generate_square_subsequent_mask(T).to(src.device)
            output = torch.zeros(src.shape[0], max_len, src.shape[-1]).type_as(src.data).to(src.device)
            output_full = torch.cat((src, output), axis=1)
            
            # auto-regressive prediction
            # input is from t to t + T_src for t < T_tgt
            # we take fwd_output[:,-1] as prediction for next pose
            for i in range(max_len):
                fwd_output = self.run_forward(output_full[:,i:(i+T)], mask)
                output_full[:,i+T] = fwd_output[:,-1].clone()
            output = output_full[:,-max_len:]
        return output
