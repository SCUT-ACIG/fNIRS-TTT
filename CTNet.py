# -*- coding: utf-8 -*-
"""
@author: iopenzd
"""
import math
import torch, copy
import torch.nn as nn
import torch.nn.functional as F

# 获取激活函数
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


#复制指定模块的实例，并返回一个模块列表
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    r"""Users may modify or implement in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()

        # d_model is emb_size 嵌入维度
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):    # 设置模型状态，检查状态中是否存在激活函数activation
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required). 到编码器层的序列
            src_mask: the mask for the src sequence (optional). SRC序列的掩码
            src_key_padding_mask: the mask for the src keys per batch (optional).  每批SRC键的掩码
        Shape:
            see the docs in Transformer class.
        """

        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)  # attn_output, attn_output_weights
        src = src + self.dropout1(src2)    # add
        src = self.norm1(src)   # norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # ffn
        src = src + self.dropout2(src2) # add
        src = self.norm2(src)  # norm
        return src, attn


class TransformerEncoder(nn.Module):
    r"""  TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required). 实例
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8) # Transformer编码器层对象，用于构建编码器的多个层
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6) # num_layers表示编码器中的层数
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, device, norm=None):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None): # 向前传播
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).      给编码器的序列
            mask: the mask for the src sequence (optional).   SRC序列的掩码
            src_key_padding_mask: the mask for the src keys per batch (optional). 每批SRC键的掩码
        Shape:
            see the docs in Transformer class.
        """

        output = src
        # 创建一个用于存储注意力权重的变量
        attn_output = torch.zeros((src.shape[1], src.shape[0], src.shape[0]), device=self.device)  # batch, seq_len, seq_len 其中batch表示批量大小，seq_len表示序列长度

        # 对每个编码器层进行迭代
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_output += attn

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output # 返回编码结果和累加的注意力权重


# absolute PE 为输入序列添加位置信息
class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__() # 定义位置编码器
        max_len = max(5000, seq_len) # 取输入序列长度seq_len和5000的最大值，用于创建位置编码矩阵的大小
        self.dropout = nn.Dropout(p=dropout) #对位置编码进行随机失活
        pe = torch.zeros(max_len, d_model) # 一个形状为(max_len, d_model)的零填充矩阵，用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 一个形状为(max_len, 1)的张量，表示位置编码的位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #一个形状为(d_model/2,)的张量，用于计算位置编码的频率因子
        pe[:, 0::2] = torch.sin(position * div_term)
        #通过使用正弦和余弦函数，将位置编码矩阵pe的偶数列填充为正弦值，奇数列填充为余弦值。这样就创建了一个能够表示序列中每个位置的位置编码矩阵

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0: -1]

        pe = pe.unsqueeze(0).transpose(0, 1) #将位置编码矩阵pe进行形状变换，使其变为形状为(1, max_len, d_model)的张量，并进行转置操作
        self.register_buffer('pe', pe) # 将位置编码矩阵pe注册为模型的缓冲区，以便在模型的前向传播中使用

    # Input:  seq_len x batch_size x dim,
    # Output: seq_len, batch_size, dim
    # 前向传播过程中，将位置编码矩阵pe的前seq_len行与输入张量x进行相加，以将位置编码添加到输入张量中的每个位置
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# contextual PE 
# class PositionalEncoding(nn.Module):
#     def __init__(self, seq_len, d_model, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout=nn.Dropout(p=dropout)
#         self.pe = nn.Conv1d(d_model, d_model, kernel_size =5, padding = 'same')
#     def forward(self, x):
#         x = x + self.pe(x)
#         return self.dropout(x)

# relative PE    
# class PositionalEncoding(nn.Module):
#     def __init__(self, seq_len, d_model, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.d_model = d_model
#         self.seq_len = seq_len

#         pe = self.generate_positional_encoding()
#         self.register_buffer('pe', pe)

#     def generate_positional_encoding(self):
#         position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
#         rel_pos = position / div_term

#         sin_rel_pos = torch.sin(rel_pos)
#         cos_rel_pos = torch.cos(rel_pos)

#         pe = torch.zeros(self.seq_len, self.d_model)
#         pe[:, 0::2] = sin_rel_pos
#         pe[:, 1::2] = cos_rel_pos

#         return pe.unsqueeze(0)

#     def forward(self, x):
#         pe = self.pe[:, :x.size(1), :]  # Adjust the positional encoding size to match the batch size of x
#         x = x + pe
#         return self.dropout(x)
    

class Permute(torch.nn.Module): # 将输入张量的维度进行置换
    def forward(self, x):
        return x.permute(1, 0)


class CTNet(nn.Module):

    def __init__(self, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_task, nlayers, dropout=0.1):
        super(CTNet, self).__init__()

        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )

        encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device)

        self.batch_norm = nn.BatchNorm1d(batch)

        # Classification Layers
        self.class_net = nn.Sequential(
            nn.Linear(emb_size, nhid_task),
            nn.ReLU(),
            Permute(),
            nn.BatchNorm1d(batch),
            Permute(),
            nn.Dropout(p=0.3),
            nn.Linear(nhid_task, nhid_task),
            nn.ReLU(),
            Permute(),
            nn.BatchNorm1d(batch),
            Permute(),
            nn.Dropout(p=0.3),
            nn.Linear(nhid_task, nclasses) # nclasses用于预测类别
        )

    def forward(self, x, task_type):
        x = self.trunk_net(x.permute(1, 0, 2))
        x, attn = self.transformer_encoder(x)
        x = self.batch_norm(x)
        # x : seq_len x batch x emb_size

        output = self.class_net(x[-1])
        return output, attn

