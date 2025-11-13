"""
CTNet: A Convolution-Transformer Network for EEG-Based Motor Imagery Classification

author: zhaowei701@163.com

"""

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pandas as pd
import random
import datetime
import time
import ttt
from pandas import ExcelWriter
# from torchsummary import summary
import torch
from torch.backends import cudnn
# from utils import calMetrics
# from utils import calculatePerClass
# from utils import numberClassChannel
import math
import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True



import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

# from utils import numberClassChannel
# from utils import load_data_evaluate
import numpy as np
import pandas as pd
from torch.autograd import Variable

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class Spatial_layer(nn.Module):#spatial attention layer
    def __init__(self):
        super(Spatial_layer, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        out = self.sigmoid(x)*identity
        return out
    
class Channel_layer(nn.Module):
    """Constructs a channel layer.
    Args:k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(Channel_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        out = x * y.expand_as(x)
        return out
    
class Temporal_layer(nn.Module):
    """Constructs a Temporal layer.
    Args:k_size: Adaptive selection of kernel size
    """
    def __init__(self, num_T=16):
        super(Temporal_layer, self).__init__()

        self.sa_layer = Spatial_layer()
        self.ch_layer = Channel_layer()

        self.conv = nn.Conv2d(2*num_T, 1*num_T, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_s = self.sa_layer(x)
        y_c = self.ch_layer(x)
        y_t = torch.cat([y_s, y_c], dim=1)
        y_t = self.conv(y_t)  

        out = self.sigmoid(y_t)
        return out

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=20, kernel_size=16, D=2, pooling_size1=2, pooling_size2=2, dropout_rate=0.3, number_channel=52, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(2, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            # Temporal_layer(num_T=f2),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            # nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 32), padding='same', bias=False), 
            # Temporal_layer(num_T=f2),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2 to adjust the length of feature into transformer encoder
            # nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )
        self.Conv1 = nn.Conv2d(2, f1, (1,1), padding='same')
        self.pre_att = Temporal_layer(num_T=f2)
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        # x = x.unsqueeze(1)
        b, _, _, _ = x.shape # b, 2, 52, 140
        # x = self.wt_conv2d(x)

        x = self.cnn_module(x)
        # print("x after cnn: ", x.shape) #(128, 40, 1, 2)
        x = self.projection(x) # (128, 2, 40)
        return x
    
class PatchEmbeddingCNN_TTT(nn.Module):
    def __init__(self, f1=20, kernel_size=16, D=2, pooling_size1=2, pooling_size2=2, dropout_rate=0.3, number_channel=52, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(2, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            # Temporal_layer(num_T=f2),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            
            # average pooling 1
            # nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 32), padding='same', bias=False), 
            Temporal_layer(num_T=f2),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2 to adjust the length of feature into transformer encoder
            # nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )
        self.pre_att = Temporal_layer(num_T=f2)
        self.projection = nn.Sequential(
            # Rearrange('b e (h) (w) -> b w (e h)'),
            Rearrange('b e (h) (w) -> b (e h) w'),
        )
        self.linear_pro = nn.Linear(128, 16) # (384, 48) (256, 32) (128, 16)
        
    def forward(self, x: Tensor) -> Tensor:
        # x = x.unsqueeze(1)
        b, _, _, _ = x.shape # b, 2, 52, 140

        x = self.cnn_module(x)
        # print("x after cnn: ", x.shape) #
        # x = x.mean(dim=1)
        # print(x.shape)
        x = self.projection(x) # 
        x = x.transpose(1, 2)
        x = self.linear_pro(x)
        x = x.transpose(1, 2)
        return x
    
class PatchEmbeddingCNN_wavelet(nn.Module):
    def __init__(self, f1=16, kernel_size=16, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=18, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(2, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            # Temporal_layer(num_T=f2),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            # Temporal_layer(num_T=f2),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )
        self.pre_att = Temporal_layer(num_T=f2)
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        # x = x.unsqueeze(1)
        b, _, _, _ = x.shape
        # print(x.shape) # [b, 1, 18, 512]
        x = self.wt_conv2d(x)

        x = self.cnn_module(x)
        x = self.projection(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer_wavelet(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN_wavelet(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
#             TransformerEncoder(heads, depth, emb_size),
        )


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
            # TransformerEncoder(heads, depth, emb_size),
        )
     
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        # x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)        
        
   
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value transformations
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)  # 可以根据需要调整 dropout 比例

    def forward(self, query, key, value):
        # query: [batch_size, seq_len1, embed_dim]
        # key: [batch_size, seq_len2, embed_dim]
        # value: [batch_size, seq_len2, embed_dim]

        batch_size = query.size(0)
        seq_len1 = query.size(1)
        seq_len2 = key.size(1)

        # Transform query, key, value
        query = self.query(query).view(batch_size, seq_len1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, seq_len2, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, seq_len2, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len1, self.embed_dim)

        return attention_output     
    
class CTNet(nn.Module):
    def __init__(self, heads=4, 
                 emb_size=16,
                 depth=6, 
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.1,
                 eeg1_number_channel = 22,
                 flatten_eeg1 = 240,
                 n_class = 2,
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = n_class, eeg1_number_channel
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        # print('self.number_channel', self.number_channel)
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1 = eeg1_f1,
                                              kernel_size = eeg1_kernel_size,
                                              D = eeg1_D,
                                              pooling_size1 = eeg1_pooling_size1,
                                              pooling_size2 = eeg1_pooling_size2,
                                              dropout_rate = eeg1_dropout_rate,
                                              )
        
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1 , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module
    def forward(self, x):
        # print(x.shape) (128, 2, 52, 140)
        cnn = self.cnn(x)
        # cnn2 = self.cnn_wavelet(x)
        #print(cnn.shape)
        #print(cnn2.shape)
        # add label 
        # cnn = cnn * math.sqrt(self.emb_size)
        # cnn = self.position(cnn) (128, 140, 40)
        
        trans = self.trans(cnn) #(128, 140, 40)

        # cnn_fusion = self.cross_attention(cnn, cnn2, cnn2)
        # cnn_fusion = self.flatten(cnn_fusion)
        
        # features = cnn2 + trans + cnn
        # features = trans + cnn_fusion
        features = trans + cnn

        # features = cnn
        # features = self.cross_attention(query=cnn, key=trans, value=trans)
        # print(features.shape)
        out = self.classification(self.flatten(features))
        return out
    
