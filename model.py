import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math
# import Simplified_TTTLinear
import ttt
# from KAN import KANLinear
from Multi_input_models import TransformerEncoderLayer
# from wtconv import WTConv2d, WTConv2d_MK2
# from ATFNet import F_Block

# conv_dw = WTConv2d(32, 32, kernel_size=5, wt_levels=3)

class FeatureExtractor:
    def __init__(self):
        self.features = None

    def save_features(self, features):
        self.features = features.detach().cpu().numpy()  # 转换为numpy方便后续t-SNE

    def get_features(self):
        return self.features


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class PreBlock(torch.nn.Module):
    """
    Preprocessing module. It is designed to replace filtering and baseline correction.

    Args:
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
    """
    def __init__(self, sampling_point):
        super().__init__()
        self.pool1 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool2 = torch.nn.AvgPool1d(kernel_size=13, stride=1, padding=6)
        self.pool3 = torch.nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.ln_0 = torch.nn.LayerNorm(sampling_point)
        self.ln_1 = torch.nn.LayerNorm(sampling_point)

    def forward(self, x):
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]

        x0 = x0.squeeze()
        x0 = self.pool1(x0)
        x0 = self.pool2(x0)
        x0 = self.pool3(x0)
        x0 = self.ln_0(x0)
        x0 = x0.unsqueeze(dim=1)

        x1 = x1.squeeze()
        x1 = self.pool1(x1)
        x1 = self.pool2(x1)
        x1 = self.pool3(x1)
        x1 = self.ln_1(x1)
        x1 = x1.unsqueeze(dim=1)

        x = torch.cat((x0, x1), 1)

        return x


class fNIRS_T(nn.Module):
    """
    fNIRS-T model

    Args:
        n_class: number of classes.
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
        dim: last dimension of output tensor after linear transformation.
        depth: number of Transformer blocks.
        heads: number of the multi-head self-attention.
        mlp_dim: dimension of the MLP layer.
        pool: MLP layer classification mode, 'cls' is [CLS] token pooling, 'mean' is  average pooling, default='cls'.
        dim_head: dimension of the multi-head self-attention, default=64.
        dropout: dropout rate, default=0.
        emb_dropout: dropout for patch embeddings, default=0.
    """
    def __init__(self, n_class, sampling_point, h, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = 100
        num_channels = 100

        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(5, 30), stride=(1, 4)),
        #     Rearrange('b c h w  -> b h (c w)'),
        #     # output width * out channels --> dim
        #     nn.Linear((math.floor((sampling_point-30)/4)+1)*8, dim),
        #     nn.LayerNorm(dim))

        # self.to_channel_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 30), stride=(1, 4)),
        #     Rearrange('b c h w  -> b h (c w)'),
        #     nn.Linear((math.floor((sampling_point-30)/4)+1)*8, dim),
        #     nn.LayerNorm(dim))

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(5, 30), stride=(1, 4)),
            Rearrange('b c h w  -> b w (c h)'),
            # output width * out channels --> dim
            nn.Linear((math.floor((h-5)/1)+1)*8, dim),
            nn.LayerNorm(dim))

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 30), stride=(1, 4)),
            Rearrange('b c h w  -> b w (c h)'),
            nn.Linear((math.floor((h-1)/1)+1)*8, dim),
            nn.LayerNorm(dim))

        self.pos_embedding_patch = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pos_embedding_channel = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.feature_extractor = FeatureExtractor()  # 初始化特征提取器
        
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, n_class))


    def forward(self, img, mask=None):
        x = self.to_patch_embedding(img)
        x2 = self.to_channel_embedding(img.squeeze())

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token_patch, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        x = self.transformer_patch(x, mask)

        b, n, _ = x2.shape

        cls_tokens = repeat(self.cls_token_channel, '() n d -> b n d', b=b)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x2 += self.pos_embedding_channel[:, :(n + 1)]
        x2 = self.dropout_channel(x2)
        x2 = self.transformer_channel(x2, mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]

        x = self.to_latent(x)
        x2 = self.to_latent(x2)
        x3 = torch.cat((x, x2), 1)
        # 保存特征
        self.feature_extractor.save_features(x3)

        def get_saved_features(self):
            return self.feature_extractor.get_features()
    
        return self.mlp_head(x3)


class fNIRS_PreT(nn.Module):
    """
    fNIRS-PreT model

    Args:
        n_class: number of classes.
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
        dim: last dimension of output tensor after linear transformation.
        depth: number of Transformer blocks.
        heads: number of the multi-head self-attention.
        mlp_dim: dimension of the MLP layer.
        pool: MLP layer classification mode, 'cls' is [CLS] token pooling, 'mean' is  average pooling, default='cls'.
        dim_head: dimension of the multi-head self-attention, default=64.
        dropout: dropout rate, default=0.
        emb_dropout: dropout for patch embeddings, default=0.
    """
    def __init__(self, n_class, sampling_point, h, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.pre = PreBlock(sampling_point)
        self.fNIRS_T = fNIRS_T(n_class, sampling_point, h, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout)

    def forward(self, img):
        img = self.pre(img)
        x = self.fNIRS_T(img)
        return x


class fNIRS_TTT_LM(nn.Module):
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', 
                 dim_head=64, dropout=0., emb_dropout=0.1,intermediate_size=4, dataset="A",
                 mini_batch_size=16, device='cpu', batch_size=128):
        super().__init__()
        self.device = device
        self.bs = batch_size
        kernel_time = 30
        match dataset:
            case "A":
                patch_num_head = 2
                input_ch = 52
                patch_channel_size = 5
                inner_channels = 8

            case "B":
                patch_num_head = 4
                input_ch = 36
                patch_channel_size = 5
                inner_channels = 8

            case "C":
                patch_num_head = 8
                input_ch = 20
                patch_channel_size = 2
                inner_channels = 8


        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=inner_channels, kernel_size=(patch_channel_size, kernel_time), padding="same"),
            # nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(2, 30), padding="same"),
            Rearrange('b c h w  -> b w (c h)'), #bs, feature_dim, ch, time -> bs, time, feature_dim*ch
            nn.Linear(inner_channels*input_ch, dim),
            nn.LayerNorm(dim)
            )

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=inner_channels, kernel_size=(1, kernel_time), padding="same"),
            Rearrange('b c h w  -> b w (c h)'), #bs, feature_dim, ch, time -> bs, time, feature_dim*ch
            nn.Linear(inner_channels*input_ch, dim),
            nn.LayerNorm(dim)
            )

        self.pos_embedding_patch = nn.Parameter(torch.randn(1, sampling_point, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.layernormalization = nn.LayerNorm(dim)
        self.cross_transformer = TransformerEncoderLayer(dim, heads)
        self.feature_extractor = FeatureExtractor()  # 初始化特征提取器

        config_p = ttt.TTTConfig(
                                hidden_size=sampling_point,           # 隐藏层大小
                                intermediate_size=sampling_point*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
                                num_hidden_layers=1,      # 隐藏层的数量
                                num_attention_heads=patch_num_head,    # 注意力头的数量
                                rms_norm_eps=1e-6,        # RMS归一化epsilon值
                                mini_batch_size=mini_batch_size )
        self.ttt_PreNorm_patch = nn.LayerNorm(dim)
        self.tttMLP = ttt.TTTMLP(config_p, layer_idx=0).to(device)
        self.patch_cache = ttt.TTTCache_MK2(self.tttMLP, batch_size, self.tttMLP.config.mini_batch_size).to(device)
        
        self.pos_embedding_channel = nn.Parameter(torch.randn(1, sampling_point, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)


        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class))
        
    def fulfill(source, aim):
        tensor1 = aim
        tensor2 = source
        padded_tensor1 = torch.cat([tensor2, torch.zeros((tensor1.shape[0] - tensor2.shape[0]), 
                                                        tensor2.shape[1], 
                                                        tensor2.shape[2],
                                                        tensor2.shape[3]).to(tensor1.device)], dim=0)
        return(padded_tensor1)

    def forward(self, img, mask=None):
        n_samples = img.shape[0]
        if n_samples != self.bs:
            img = torch.cat([img, torch.zeros((self.bs - n_samples, 
                                               img.shape[1],
                                               img.shape[2],
                                               img.shape[3])).to(self.device)], dim=0)
        x = self.to_patch_embedding(img)
        x2 = self.to_channel_embedding(img.squeeze())

        # pos embedding
        b, n, _ = x.shape
        x += self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        b, n, _ = x2.shape
        x2 += self.pos_embedding_channel[:, :(n + 1)]
        x2 = self.dropout_channel(x2)
        #cross attn
        cr_s1 = x
        cr_s2 = x2
        x = self.cross_transformer(cr_s1, cr_s2, cr_s2)

        #ttt
        xres = x = self.ttt_PreNorm_patch(x)
        x = x.transpose(1, 2)
        patch_position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
        x = self.tttMLP(x, position_ids=patch_position_ids ,cache_params=self.patch_cache)
        x = x.transpose(1, 2)
        x = x + xres

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x_cross = self.to_latent(x)

        # 保存特征
        self.feature_extractor.save_features(x_cross)

        def get_saved_features(self):
            return self.feature_extractor.get_features()
        
        return self.mlp_head(x_cross)[:n_samples]



class fNIRS_TTT_M(nn.Module):
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', 
                 dim_head=64, dropout=0., emb_dropout=0.1,intermediate_size=4, dataset="A",
                 mini_batch_size=16, device='cpu', batch_size=128):
        super().__init__()
        self.device = device
        self.bs = batch_size
        kernel_time = 30
        match dataset:
            case "A":
                patch_num_head = 2
                input_ch = 52
                patch_channel_size = 5
                inner_channels = 8

            case "B":
                patch_num_head = 4
                input_ch = 36
                patch_channel_size = 5
                inner_channels = 8

            case "C":
                patch_num_head = 8
                input_ch = 20
                patch_channel_size = 2
                inner_channels = 8


        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=inner_channels, kernel_size=(patch_channel_size, kernel_time), padding="same"),
            # nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(2, 30), padding="same"),
            Rearrange('b c h w  -> b w (c h)'), #bs, feature_dim, ch, time -> bs, time, feature_dim*ch
            nn.Linear(inner_channels*input_ch, dim),
            nn.LayerNorm(dim)
            )

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=inner_channels, kernel_size=(1, kernel_time), padding="same"),
            Rearrange('b c h w  -> b w (c h)'), #bs, feature_dim, ch, time -> bs, time, feature_dim*ch
            nn.Linear(inner_channels*input_ch, dim),
            nn.LayerNorm(dim)
            )

        self.pos_embedding_patch = nn.Parameter(torch.randn(1, sampling_point, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.layernormalization = nn.LayerNorm(dim)
        self.cross_transformer = TransformerEncoderLayer(dim, heads)
        self.feature_extractor = FeatureExtractor()  # 初始化特征提取器

        config_p = ttt.TTTConfig(
                                hidden_size=sampling_point,           # 隐藏层大小
                                intermediate_size=sampling_point*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
                                num_hidden_layers=1,      # 隐藏层的数量
                                num_attention_heads=patch_num_head,    # 注意力头的数量
                                rms_norm_eps=1e-6,        # RMS归一化epsilon值
                                mini_batch_size=mini_batch_size )
        self.ttt_PreNorm_patch = nn.LayerNorm(dim)
        self.tttMLP = ttt.TTTMLP(config_p, layer_idx=0).to(device)
        self.patch_cache = ttt.TTTCache_MK2(self.tttMLP, batch_size, self.tttMLP.config.mini_batch_size).to(device)
        
        self.pos_embedding_channel = nn.Parameter(torch.randn(1, sampling_point, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)


        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class))
        
    def fulfill(source, aim):
        tensor1 = aim
        tensor2 = source
        padded_tensor1 = torch.cat([tensor2, torch.zeros((tensor1.shape[0] - tensor2.shape[0]), 
                                                        tensor2.shape[1], 
                                                        tensor2.shape[2],
                                                        tensor2.shape[3]).to(tensor1.device)], dim=0)
        return(padded_tensor1)

    def forward(self, img, mask=None):
        n_samples = img.shape[0]
        if n_samples != self.bs:
            img = torch.cat([img, torch.zeros((self.bs - n_samples, 
                                               img.shape[1],
                                               img.shape[2],
                                               img.shape[3])).to(self.device)], dim=0)
        x = self.to_patch_embedding(img)
        x2 = self.to_channel_embedding(img.squeeze())

        # pos embedding
        b, n, _ = x.shape
        x += self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        b, n, _ = x2.shape
        x2 += self.pos_embedding_channel[:, :(n + 1)]
        x2 = self.dropout_channel(x2)
        #cross attn
        cr_s1 = x
        cr_s2 = x2
        x = self.cross_transformer(cr_s1, cr_s2, cr_s2)

        #ttt
        x = self.ttt_PreNorm_patch(x)
        x = x.transpose(1, 2)
        patch_position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
        x = self.tttMLP(x, position_ids=patch_position_ids ,cache_params=self.patch_cache)
        x = x.transpose(1, 2)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x_cross = self.to_latent(x)

        # 保存特征
        self.feature_extractor.save_features(x_cross)

        def get_saved_features(self):
            return self.feature_extractor.get_features()
        
        return self.mlp_head(x_cross)[:n_samples]


class fNIRS_TTT_LL(nn.Module):
    def __init__(self, n_class, sampling_point, h, dim, depth, heads, mlp_dim, pool='cls', 
                 dim_head=64, dropout=0., emb_dropout=0.,intermediate_size=4, dataset="A",
                 mini_batch_size=16, device='cpu', batch_size=128):
        super().__init__()
        self.device = device
        self.bs = batch_size
        num_patches = 100
        num_channels = 100
        match dataset:
            case "A":
                patch_hiden_size = 28
                patch_num_head = 2
                channel_hiden_size = 28
                channel_num_head = 2
            case "B":
                patch_hiden_size = 42
                patch_num_head = 1
                channel_hiden_size = 42
                channel_num_head = 1
            case "C":
                patch_hiden_size = 56
                patch_num_head = 2
                channel_hiden_size = 56
                channel_num_head = 2

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(5, 30), stride=(1, 4)),
            Rearrange('b c h w  -> b w (c h)'),
            # output width * out channels --> dim
            nn.Linear((math.floor((h-5)/1)+1)*8, dim),
            nn.LayerNorm(dim))

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 30), stride=(1, 4)),
            Rearrange('b c h w  -> b w (c h)'),
            nn.Linear((math.floor((h-1)/1)+1)*8, dim),
            nn.LayerNorm(dim))

        self.pos_embedding_patch = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.feature_extractor = FeatureExtractor()  # 初始化特征提取器

        # 加入的ttt patch
        # config_p = ttt.TTTConfig(
        #                         hidden_size=patch_hiden_size,           # 隐藏层大小
        #                         intermediate_size=patch_hiden_size*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
        #                         num_hidden_layers=1,      # 隐藏层的数量
        #                         num_attention_heads=patch_num_head,    # 注意力头的数量
        #                         rms_norm_eps=1e-6,        # RMS归一化epsilon值
        #                         mini_batch_size=mini_batch_size )
        config_p = ttt.TTTConfig(
                                hidden_size=patch_hiden_size,           # 隐藏层大小
                                intermediate_size=patch_hiden_size*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
                                num_hidden_layers=1,      # 隐藏层的数量
                                num_attention_heads=patch_num_head,    # 注意力头的数量
                                rms_norm_eps=1e-6,        # RMS归一化epsilon值
                                mini_batch_size=mini_batch_size )
        self.ttt_PreNorm_patch = nn.LayerNorm(dim)
        self.tttLinear_patch = ttt.TTTLinear(config_p, layer_idx=0).to(device)
        self.patch_cache = ttt.TTTCache_MK2(self.tttLinear_patch, batch_size, self.tttLinear_patch.config.mini_batch_size).to(device)
        

        self.pos_embedding_channel = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # 加入的ttt channel
        config_c = ttt.TTTConfig(
                                hidden_size=channel_hiden_size,           # 隐藏层大小
                                intermediate_size=channel_hiden_size*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
                                num_hidden_layers=1,      # 隐藏层的数量
                                num_attention_heads=channel_num_head,    # 注意力头的数量
                                rms_norm_eps=1e-6,        # RMS归一化epsilon值
                                mini_batch_size=mini_batch_size )
        self.ttt_PreNorm_channel = nn.LayerNorm(dim)
        self.tttLinear_channel = ttt.TTTLinear(config_c, layer_idx=0).to(device)
        self.channel_cache = ttt.TTTCache_MK2(self.tttLinear_channel, batch_size, self.tttLinear_channel.config.mini_batch_size).to(device)
        

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, n_class))
        
    def fulfill(source, aim):
        tensor1 = aim
        tensor2 = source
        padded_tensor1 = torch.cat([tensor2, torch.zeros((tensor1.shape[0] - tensor2.shape[0]), 
                                                        tensor2.shape[1], 
                                                        tensor2.shape[2],
                                                        tensor2.shape[3]).to(tensor1.device)], dim=0)
        return(padded_tensor1)

    def forward(self, img, mask=None):
        n_samples = img.shape[0]
        if n_samples != self.bs:
            img = torch.cat([img, torch.zeros((self.bs - n_samples, 
                                               img.shape[1],
                                               img.shape[2],
                                               img.shape[3])).to(self.device)], dim=0)
        x = self.to_patch_embedding(img)
        x2 = self.to_channel_embedding(img.squeeze())


        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token_patch, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        # x = self.transformer_patch(x, mask)

        x = x[:,1:,:]
        if n % 2 != 0:
            x = x[:, 1:,:]

        x = self.ttt_PreNorm_patch(x)
        x = x.transpose(1, 2)
        print(x.shape)
        patch_position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
        x = self.tttLinear_patch(x, position_ids=patch_position_ids ,cache_params=self.patch_cache)
        x = x.transpose(1, 2)

        b, n, _ = x2.shape


        cls_tokens = repeat(self.cls_token_channel, '() n d -> b n d', b=b)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x2 += self.pos_embedding_channel[:, :(n + 1)]
        x2 = self.dropout_channel(x2)
        # x2 = self.transformer_channel(x2, mask)

        x2 = x2[:,1:,:]
        if n % 2 != 0:
            x2 = x2[:, 1:,:]
        x2 = self.ttt_PreNorm_channel(x2)
        x2 = x2.transpose(1, 2)
        channel_position_ids = torch.arange(x2.shape[1]).unsqueeze(0).repeat(x2.shape[0],1).to(self.device)
        x2 = self.tttLinear_channel(x2, position_ids=channel_position_ids ,cache_params=self.channel_cache)
        x2 = x2.transpose(1, 2)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]

        x = self.to_latent(x)
        x2 = self.to_latent(x2)
        x3 = torch.cat((x, x2), 1)

        self.feature_extractor.save_features(x3)

        def get_saved_features(self):
            return self.feature_extractor.get_features()
        
        del channel_position_ids, patch_position_ids
        return self.mlp_head(x3)[:n_samples]


class fNIRS_TTT_L(nn.Module):
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', 
                 dim_head=64, dropout=0., emb_dropout=0.1,intermediate_size=4, dataset="A",
                 mini_batch_size=16, device='cpu', batch_size=128):
        super().__init__()
        self.device = device
        self.bs = batch_size
        kernel_time = 30
        match dataset:
            case "A":
                patch_num_head = 2
                channel_num_head = 2
                input_ch = 52
                kernel_size = (5,kernel_time)
            case "B":
                patch_num_head = 4
                channel_num_head = 2
                input_ch = 36
                kernel_size = (5,kernel_time)
            case "C":
                patch_num_head = 4
                channel_num_head = 2
                input_ch = 20
                kernel_size = (2,kernel_time)

        self.to_patch_embedding = nn.Sequential(
            # nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(5, 30), padding="same"),
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=kernel_size, padding="same"),
            Rearrange('b c h w  -> b w (c h)'), #bs, feature_dim, ch, time -> bs, time, feature_dim*ch
            nn.Linear(8*input_ch, dim),
            nn.LayerNorm(dim)
            )

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, kernel_time), padding="same"),
            Rearrange('b c h w  -> b w (c h)'), #bs, feature_dim, ch, time -> bs, time, feature_dim*ch
            nn.Linear(8*input_ch, dim),
            nn.LayerNorm(dim)
            )

        self.pos_embedding_patch = nn.Parameter(torch.randn(1, sampling_point, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.layernormalization = nn.LayerNorm(dim)
        self.cross_transformer = TransformerEncoderLayer(dim, heads)
        self.feature_extractor = FeatureExtractor()  # 初始化特征提取器

        config_p = ttt.TTTConfig(
                                hidden_size=sampling_point,           # 隐藏层大小
                                intermediate_size=sampling_point*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
                                num_hidden_layers=1,      # 隐藏层的数量
                                num_attention_heads=patch_num_head,    # 注意力头的数量
                                rms_norm_eps=1e-6,        # RMS归一化epsilon值
                                mini_batch_size=mini_batch_size )
        self.ttt_PreNorm_patch = nn.LayerNorm(dim)
        self.tttMLP_patch = ttt.TTTLinear(config_p, layer_idx=0).to(device)
        self.patch_cache = ttt.TTTCache_MK2(self.tttMLP_patch, batch_size, self.tttMLP_patch.config.mini_batch_size).to(device)
        

        self.pos_embedding_channel = nn.Parameter(torch.randn(1, sampling_point, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        # 加入的ttt channel
        config_c = ttt.TTTConfig(
                                hidden_size=sampling_point,           # 隐藏层大小
                                intermediate_size=sampling_point*intermediate_size,    # MLP中间层的大小，可以设置为hidden_size的倍数
                                num_hidden_layers=1,      # 隐藏层的数量
                                num_attention_heads=channel_num_head,    # 注意力头的数量
                                rms_norm_eps=1e-6,        # RMS归一化epsilon值
                                mini_batch_size=mini_batch_size )
        self.ttt_PreNorm_channel = nn.LayerNorm(dim)
        self.tttMLP_channel = ttt.TTTLinear(config_c, layer_idx=0).to(device)
        self.channel_cache = ttt.TTTCache_MK2(self.tttMLP_channel, batch_size, self.tttMLP_channel.config.mini_batch_size).to(device)
        

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class))
        
    def fulfill(source, aim):
        tensor1 = aim
        tensor2 = source
        padded_tensor1 = torch.cat([tensor2, torch.zeros((tensor1.shape[0] - tensor2.shape[0]), 
                                                        tensor2.shape[1], 
                                                        tensor2.shape[2],
                                                        tensor2.shape[3]).to(tensor1.device)], dim=0)
        return(padded_tensor1)

    def forward(self, img, mask=None):
        n_samples = img.shape[0]
        if n_samples != self.bs:
            img = torch.cat([img, torch.zeros((self.bs - n_samples, 
                                               img.shape[1],
                                               img.shape[2],
                                               img.shape[3])).to(self.device)], dim=0)
        x = self.to_patch_embedding(img)
        x2 = self.to_channel_embedding(img.squeeze())

        # pos embedding
        b, n, _ = x.shape
        x += self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        b, n, _ = x2.shape
        x2 += self.pos_embedding_channel[:, :(n + 1)]
        x2 = self.dropout_channel(x2)
        #cross attn
        cr_s1 = x
        cr_s2 = x2
        x = self.cross_transformer(cr_s1, cr_s2, cr_s2)

        #ttt
        x = self.ttt_PreNorm_patch(x)
        x = x.transpose(1, 2)
        patch_position_ids = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
        x = self.tttMLP_patch(x, position_ids=patch_position_ids ,cache_params=self.patch_cache)
        x = x.transpose(1, 2)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x_cross = self.to_latent(x)

        # 保存特征
        self.feature_extractor.save_features(x_cross)

        def get_saved_features(self):
            return self.feature_extractor.get_features()
        
        return self.mlp_head(x_cross)[:n_samples]
