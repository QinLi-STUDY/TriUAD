"""
Added get selfattention from all layer

Mostly copy-paster from DINO (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
and timm library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

"""
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.nn import functional as F
import torch
import torch.nn as nn




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



# class Prototype_Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         # self.scale = qk_scale or head_dim ** -0.5
#         self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
#         # Query takes concatenated features [x, x_seg, x_ske] => 3 * dim
#         self.q = nn.Linear(dim * 3, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

    # def forward(self, x, x_seg, x_ske, agg_prototype):
    #     """
    #     :param x: 输入特征图 [B, 784, 768]
    #     :param x_seg: 辅助特征图 [B, 784, 768]，用于增强 x 的表达
    #     :param x_ske: 辅助特征图 [B, 784, 768]，用于增强 x 的表达
    #     :param agg_prototype: 聚合原型 [B, 6, 768]
    #     :return: 输出 x 和注意力矩阵 attn
    #     """
    #     B, N, C = x.shape  # N=784, C=768
    #     prototype_num = agg_prototype.shape[1]  # 6

    #     # 1) 融合 x, x_seg 和 x_ske（在通道维度上拼接）
    #     x_combined = torch.cat([x, x_seg, x_ske], dim=-1)  # [B, 784, 3*C]

    #     # 2) 计算查询（q），键（k），值（v）
    #     q = self.q(x_combined).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]  # [1, B, num_heads, N, head_dim]
    #     kv = self.kv(agg_prototype).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, B, num_heads, prototype_num, head_dim]
    #     k, v = kv[0], kv[1]  # k, v shapes: [B, num_heads, prototype_num, head_dim]

    #     # 3) 正规化查询（q）和键（k）
    #     q = torch.nn.functional.normalize(q, dim=-1)
    #     k = torch.nn.functional.normalize(k, dim=-1)

    #     # 4) 计算注意力权重
    #     attn = (q @ k.transpose(-2, -1)) * self.learn_scale  # [B, num_heads, N, prototype_num]
    #     attn = F.relu(attn)
    #     attn = self.attn_drop(attn)

        # # 5) 加权求和
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # return x, attn


# Intensity Enhancement Layer
class IEL(nn.Module): # 强度增强层
    def __init__(self, dim, ffn_expansion_factor=1, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Linear(dim, hidden_features * 2, bias=bias)

        self.dwconv = nn.Linear(hidden_features * 2, hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Linear(hidden_features, hidden_features, bias=bias)
        self.dwconv2 = nn.Linear(hidden_features, hidden_features, bias=bias)

        self.project_out = nn.Linear(hidden_features, dim, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        # 保存原始形状
        B,N,C = x.shape# 16,768,784
        x = self.project_in(x)# [16, 784, 4084]
        x1,x2 = self.dwconv(x).chunk(2, dim=-1) # [16, 784, 2042]
        x1 = self.Tanh(self.dwconv1(x1)) + x1# [16, 784, 2042]
        x2 = self.Tanh(self.dwconv2(x2)) + x2# [16, 784, 2042]
        x = x1 * x2# [16, 784, 2042]
        x = self.project_out(x)# [16, 784, 4084]
        
        return x

class Prototype_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        
        # self.learn_scale_seg = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        # self.learn_scale_ske = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        # self.kv_seg = nn.Linear(dim, dim *2, bias=qkv_bias)
        # self.kv_ske = nn.Linear(dim, dim *2, bias=qkv_bias)
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim *2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # self.iel = IEL(dim=dim)
        
        self.fusion_weight_seg = nn.Parameter(torch.ones(1), requires_grad=True)
        self.fusion_weight_ske = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x, prototype_token):
        B, N, C = x.shape
        prototype_num = prototype_token.shape[1]
        # prototype_token = prototype_token.repeat((B, 1, 1))
        # q, k, b [B, head_num, num_token, C]
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0] # [2, 12, 784, 64]
        kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)# [2, 2, 12, 6, 64]
        k, v = kv[0], kv[1] # [2, 12, 6, 64] [2, 12, 6, 64]
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale# [2, 12, 784, 6]
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # x = self.iel(x)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
    # def forward(self, x, prototype_token, segment_token, sketch_token):
    #     B, N, C = x.shape
    #     prototype_num = prototype_token.shape[1]
    #     # prototype_token = prototype_token.repeat((B, 1, 1))
    #     # q, k, b [B, head_num, num_token, C]
    #     q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0] # [2, 12, 784, 64]
    #     kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)# [2, 2, 12, 6, 64]
    #     kv_seg = self.kv(segment_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)# [2, 2, 12, 6, 64]
    #     kv_ske = self.kv(sketch_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)# [2, 2, 12, 6, 64]
    #     k_seg, v_seg = kv_seg[0], kv_seg[1]
    #     k_ske, v_ske = kv_ske[0], kv_ske[1]
    #     k_seg = torch.nn.functional.normalize(k_seg, dim=-1)
    #     k_ske = torch.nn.functional.normalize(k_ske, dim=-1)
    #     attn_seg = (q @ k_seg.transpose(-2, -1)) * self.learn_scale_seg
    #     attn_seg = F.relu(attn_seg)
    #     attn_seg = self.attn_drop(attn_seg)
    #     x_seg = (attn_seg @ v_seg).transpose(1, 2).reshape(B, N, C)
    #     attn_ske = (q @ k_ske.transpose(-2, -1)) * self.learn_scale_ske
    #     attn_ske = F.relu(attn_ske)
    #     attn_ske = self.attn_drop(attn_ske)
    #     x_ske = (attn_ske @ v_ske).transpose(1, 2).reshape(B, N, C)
        
        # k, v = kv[0], kv[1] # [2, 12, 6, 64] [2, 12, 6, 64]
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        # attn = (q @ k.transpose(-2, -1)) * self.learn_scale# [2, 12, 784, 6]
        # attn = F.relu(attn)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # x = x + self.fusion_weight_seg * x_seg + self.fusion_weight_ske * x_ske
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x, attn
    

class Aggregation_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, T, C = x.shape
        _, N, _ = y.shape
        q = self.q(x).reshape(B, T, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attnmap = attn.softmax(dim=-1)
        attn = self.attn_drop(attnmap)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Aggregation_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Aggregation_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
# from models.CFEM import CFEM
# from models.CrossEnhanceModule import CFEM
class Prototype_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = CFEM(in_channels=dim,num_heads=num_heads,qkv_bias=qkv_bias)
        self.attn = Prototype_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # def forward(self, x,segment_feature, sketch_feature, prototype, return_attention=False):
    #     # if attn_mask is not None:
    #     #     y, attn = self.attn(self.norm1(x))
    #     # else:
    #     y, attn = self.attn(self.norm1(x), self.norm1(segment_feature), self.norm1(sketch_feature), self.norm1(prototype))
    #     x = self.drop_path(y)
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     if return_attention:
    #         return x, attn
    #     else:
    #         return x
    
    def forward(self, x, prototype, return_attention=False):
        # if attn_mask is not None:
        #     y, attn = self.attn(self.norm1(x))
        # else:
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x
    # def forward(self, x, prototype, segment_token, sketch_token, return_attention=False):
    #     # if attn_mask is not None:
    #     #     y, attn = self.attn(self.norm1(x))
    #     # else:
    #     y, attn = self.attn(self.norm1(x), self.norm1(prototype), self.norm1(segment_token), self.norm1(sketch_token))
    #     x = self.drop_path(y)
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     if return_attention:
    #         return x, attn
    #     else:
    #         return x





