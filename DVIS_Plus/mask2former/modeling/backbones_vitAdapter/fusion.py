# fusion.py
# 完整实现三种跨模态融合模块：CMX、TokenFusion、CEN
# 设计参考原论文，支持 2D 特征图 (B, C, H, W) 输入，也支持 token 序列 (B, N, C) 通过 reshape 转换
# 包含必要的 LayerScale、DropPath、Mlp 等辅助类

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math


# ----------------------------------------------------------------------
# 辅助模块
# ----------------------------------------------------------------------
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
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


class LayerScale(nn.Module):
    """Layer scale module, as used in CaiT (Going deeper with Image Transformers)"""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


# ----------------------------------------------------------------------
# 1. CMX 融合
# ----------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Cross-attention module: query from one modality, key/value from another."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, N, C = query.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMXBlock(nn.Module):
    """
    CMX block containing self-attention (optional) and cross-attention for two modalities.
    This is a faithful implementation of the cross-modal fusion block from the CMX paper.
    Each modality has its own self-attention and FFN, and cross-attention is performed in between.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-5):
        super().__init__()
        self.norm1_rgb = norm_layer(dim)
        self.norm1_depth = norm_layer(dim)
        self.self_attn_rgb = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.self_attn_depth = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.cross_attn_rgb = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_depth = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path1_rgb = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1_depth = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2_rgb = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2_depth = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3_rgb = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3_depth = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_rgb = norm_layer(dim)
        self.norm2_depth = norm_layer(dim)
        self.norm3_rgb = norm_layer(dim)
        self.norm3_depth = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_rgb = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # LayerScale (optional, can be omitted if init_values <= 0)
        self.ls1_rgb = LayerScale(dim, init_values) if init_values > 0 else nn.Identity()
        self.ls1_depth = LayerScale(dim, init_values) if init_values > 0 else nn.Identity()
        self.ls2_rgb = LayerScale(dim, init_values) if init_values > 0 else nn.Identity()
        self.ls2_depth = LayerScale(dim, init_values) if init_values > 0 else nn.Identity()
        self.ls3_rgb = LayerScale(dim, init_values) if init_values > 0 else nn.Identity()
        self.ls3_depth = LayerScale(dim, init_values) if init_values > 0 else nn.Identity()

    def forward(self, rgb, depth):
        # rgb, depth: token sequences of shape (B, N, C)
        # 1. Self-attention
        rgb_self, _ = self.self_attn_rgb(self.norm1_rgb(rgb), self.norm1_rgb(rgb), self.norm1_rgb(rgb))
        rgb = rgb + self.drop_path1_rgb(self.ls1_rgb(rgb_self))
        depth_self, _ = self.self_attn_depth(self.norm1_depth(depth), self.norm1_depth(depth), self.norm1_depth(depth))
        depth = depth + self.drop_path1_depth(self.ls1_depth(depth_self))

        # 2. Cross-attention
        rgb_cross = self.cross_attn_rgb(self.norm2_rgb(rgb), self.norm2_depth(depth), self.norm2_depth(depth))
        rgb = rgb + self.drop_path2_rgb(self.ls2_rgb(rgb_cross))
        depth_cross = self.cross_attn_depth(self.norm2_depth(depth), self.norm2_rgb(rgb), self.norm2_rgb(rgb))
        depth = depth + self.drop_path2_depth(self.ls2_depth(depth_cross))

        # 3. FFN
        rgb_ffn = self.mlp_rgb(self.norm3_rgb(rgb))
        rgb = rgb + self.drop_path3_rgb(self.ls3_rgb(rgb_ffn))
        depth_ffn = self.mlp_depth(self.norm3_depth(depth))
        depth = depth + self.drop_path3_depth(self.ls3_depth(depth_ffn))

        return rgb, depth


class CMXFusion(nn.Module):
    """
    CMX fusion module for 2D feature maps.
    Input: rgb (B, C, H, W), depth (B, C, H, W)
    Output: fused rgb feature (B, C, H, W)  (we return only rgb branch as primary)
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., init_values=1e-5):
        super().__init__()
        self.block = CMXBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False,
                              drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                              act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=init_values)

    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape
        rgb_flat = rgb.flatten(2).transpose(1, 2)      # (B, HW, C)
        depth_flat = depth.flatten(2).transpose(1, 2)  # (B, HW, C)
        rgb_out, depth_out = self.block(rgb_flat, depth_flat)
        rgb_out = rgb_out.transpose(1, 2).view(B, C, H, W)
        depth_out = depth_out.transpose(1, 2).view(B, C, H, W)
        return rgb_out + depth_out


# ----------------------------------------------------------------------
# 2. TokenFusion 融合
# ----------------------------------------------------------------------
class TokenFusion(nn.Module):
    """
    TokenFusion module as described in the paper.
    It dynamically selects tokens from either modality using a light-weight mask generator.
    The mask is generated from the concatenated features and is applied to choose
    between RGB and depth tokens at each spatial position.

    For training, a sigmoid function is used to produce soft masks. During inference,
    the mask can be binarized with a threshold (default 0.5) or left soft.
    """
    def __init__(self, dim, reduction=4, hard=False):
        super().__init__()
        self.hard = hard
        self.mask_generator = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 1, kernel_size=1, bias=True)
        )

    def forward(self, rgb, depth):
        concat = torch.cat([rgb, depth], dim=1)               # (B, 2C, H, W)
        logits = self.mask_generator(concat)                  # (B, 1, H, W)

        if self.hard:
            # Straight-through estimator for hard selection
            mask = torch.sigmoid(logits)
            hard_mask = (mask > 0.5).float()
            mask = hard_mask - mask.detach() + mask
        else:
            mask = torch.sigmoid(logits)

        return rgb * mask + depth * (1 - mask)


# ----------------------------------------------------------------------
# 3. CEN 融合
# ----------------------------------------------------------------------
class CENFusion(nn.Module):
    """
    Cross-modal Enhancement Network (CEN) fusion module.
    Implements both spatial and channel attention to enhance RGB features using depth features,
    and also enhances depth features using RGB features (bidirectional enhancement).
    The two enhanced branches are then combined.

    This implementation follows the CEN paper's core idea: cross-modal spatial/channel attention.
    """
    def __init__(self, dim, reduction=16, init_values=1e-5):
        super().__init__()
        # Spatial attention for RGB from depth
        self.spatial_att_rgb = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # Channel attention for RGB from depth
        self.channel_att_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention for depth from RGB
        self.spatial_att_depth = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # Channel attention for depth from RGB
        self.channel_att_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.gamma_rgb = nn.Parameter(init_values * torch.ones(1))
        self.gamma_depth = nn.Parameter(init_values * torch.ones(1))

    def forward(self, rgb, depth):
        # Enhance RGB using depth
        rgb_spatial = self.spatial_att_rgb(depth)       # (B,1,H,W)
        rgb_channel = self.channel_att_rgb(depth)       # (B,C,1,1)
        rgb_enhanced = rgb * rgb_spatial * rgb_channel
        rgb_out = rgb + self.gamma_rgb * rgb_enhanced

        # Enhance depth using RGB
        depth_spatial = self.spatial_att_depth(rgb)     # (B,1,H,W)
        depth_channel = self.channel_att_depth(rgb)     # (B,C,1,1)
        depth_enhanced = depth * depth_spatial * depth_channel
        depth_out = depth + self.gamma_depth * depth_enhanced

        # Final fusion: we can return rgb_out (primary) or combine both
        # Here we return rgb_out to maintain compatibility with other fusion layers
        return rgb_out+depth_out


# ----------------------------------------------------------------------
# 工厂函数
# ----------------------------------------------------------------------
def build_fusion_layer(fusion_type, dim, **kwargs):
    if fusion_type == 'cmx':
        return CMXFusion(dim, **kwargs)
    elif fusion_type == 'tokenfusion':
        return TokenFusion(dim, **kwargs)
    elif fusion_type == 'cen':
        return CENFusion(dim, **kwargs)
    elif fusion_type == 'none' or fusion_type is None:
        return None
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")