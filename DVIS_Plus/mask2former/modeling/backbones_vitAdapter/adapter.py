# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

import logging
from functools import partial

import torch.utils.checkpoint as cp
from timm.models.layers import DropPath

from .backbones import get_models
from .fusion import build_fusion_layer
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls

class InteractionBlockWithCls_Efficient(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def foward_vit_blocks(self, blocks, x, windows_nums=3):
        bs = x.shape[0]
        nums = bs // windows_nums
        if bs != windows_nums * nums:
            nums += 1

        ret = []
        for i in range(nums):
            start = i * windows_nums
            end = min(start + windows_nums, bs)
            x_ = x[start: end]
            for idx, blk in enumerate(blocks):
                x_ = blk(x_)
            ret.append(x_)

        del x
        return torch.cat(ret, dim=0)

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W, finetune=False):
        if finetune:
            x = torch.cat((cls, x), dim=1)
            for idx, blk in enumerate(blocks):
                x = blk(x)
            cls, x = x[:, :1, ], x[:, 1:, ]
        else:
            with torch.no_grad():
                x = torch.cat((cls, x), dim=1)
                for idx, blk in enumerate(blocks):
                    x = blk(x)
                cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, in_channels=3, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

def get_adapter_args(name='vitl', backbone_weight=None,
                     freeze_backbone=False, finetune=False,
                     finetune_indexes=[0, ], with_cp=False, 
                     fusion_type='none', depth_in_channels=1):
    if freeze_backbone:
        assert backbone_weight is not None
    vit_backbone = get_models(name, backbone_weight)
    bg_configs = vit_backbone.configs_dict
    if name == 'vitl':
        adapter_args = {
            'vit_module': vit_backbone,
            'pretrain_size': bg_configs['img_size'],
            'init_values': 1e-6,
            'conv_inplane': 64,
            'n_points': 4,
            'deform_num_heads': 16,
            'with_cffn': True,
            'cffn_ratio': 0.25,
            'deform_ratio': 0.5,
            'add_vit_feature': True,
            'use_extra_extractor': True,
            'with_cp': with_cp,
            'interaction_indexes': [[0, 5], [6, 11], [12, 17], [18, 23]],
        }
    elif name == 'vitb':
        adapter_args = {
            'vit_module': vit_backbone,
            'pretrain_size': bg_configs['img_size'],
            'init_values': 1e-6,
            'conv_inplane': 64,
            'n_points': 4,
            'deform_num_heads': 12,
            'with_cffn': True,
            'cffn_ratio': 0.25,
            'deform_ratio': 0.5,
            'add_vit_feature': True,
            'use_extra_extractor': True,
            'with_cp': with_cp,
            'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
        }
    else:
        raise NotImplementedError
    adapter_args.update({
        'fusion_type': fusion_type,
        'depth_in_channels': depth_in_channels,
        'freeze_backbone': freeze_backbone,
        'finetune': finetune,
        'finetune_indexes': finetune_indexes,
    })
    #adapter_model = DinoV2ViTAdapter(**adapter_args)
    return adapter_args

class DinoV2ViTAdapter(nn.Module):
    def __init__(
        self,
        vit_module=None,
        pretrain_size=224,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
        freeze_backbone=False,
        finetune=False,
        finetune_indexes=[0, ],
        fusion_type=None, 
        depth_in_channels=None, 
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.depth_in_channels = depth_in_channels 
        self.num_block = len(vit_module.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = vit_module.embed_dim

        self.drop_path_rate = vit_module.configs_dict['drop_path_rate']
        self.norm_layer = vit_module.norm_layer

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            # InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
            InteractionBlockWithCls_Efficient(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
            #InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        self.vit_module = vit_module
        if freeze_backbone:
            for p in self.vit_module.parameters():
                p.requires_grad_(False)
            if finetune:
                # self.vit_module.patch_embed.requires_grad_(True)
                # self.vit_module.cls_token.requires_grad_(True)
                # self.vit_module.pos_embed.requires_grad_(True)
                finetuned_interaction_indexes = finetune_indexes
                self.finetuned_interaction_indexes = finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    for p in self.vit_module.blocks[indexes[0]:indexes[-1] + 1].parameters():
                        p.requires_grad_(True)

        self.freeze_backbone = freeze_backbone
        self.finetune = finetune

        if not hasattr(self, "finetuned_interaction_indexes"):
            self.finetuned_interaction_indexes = list(range(len(self.interaction_indexes)))

        # 如果正处于全参微调模式下（FREEZE_VIT=False），
        # 我们必须手动关闭未使用的 norm 和 mask_token 的梯度属性（requires_grad=False），
        # 从而彻底避开 DDP 报错，同时保障 Patch_embed, Pos_embed, CLS 等全参数能被完整微调训练！
        if not self.freeze_backbone:
            # A. 手动关闭未使用的 2D 归一化层梯度
            if hasattr(self.vit_module, "norm") and self.vit_module.norm is not None:
                for p in self.vit_module.norm.parameters():
                    p.requires_grad = False
            # B. 手动关闭未使用的 MIM 掩膜 Token 梯度
            if hasattr(self.vit_module, "mask_token") and self.vit_module.mask_token is not None:
                self.vit_module.mask_token.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                # self.vit_module.patch_embed.train()
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, finetune=finetune)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONE_REGISTRY.register()
class D2VitAdapterDinoV2(DinoV2ViTAdapter, Backbone):
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        super().__init__(**adapter_args)

        self._out_features = ['res2', 'res3', 'res4', 'res5']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        # self._out_feature_channels = {
        #     "res2": self.num_features[0],
        #     "res3": self.num_features[1],
        #     "res4": self.num_features[2],
        #     "res5": self.num_features[3],
        # }
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

from research.modeling.loftup_utils.upsamplers.upsamplers import load_loftup_checkpoint

class DinoV2ViTAdapter_LoftUP(DinoV2ViTAdapter):
    def __init__(self, *args, **kwargs):
        # 🟢 1. 在调用 super().__init__ 之前拦截 cfg 变量，防止父类抛出 TypeError
        cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)

        # ==========================================
        # 🟢 2. 注册 DVIS 的均值/方差以及通道格式（防御性对齐所需的 Buffer）
        # ==========================================
        if cfg is not None:
            self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1), False)
            self.input_format = cfg.INPUT.FORMAT
        else:
            # 兜底默认 ImageNet 255级配置
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.280, 103.530]).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.120, 57.375]).view(1, -1, 1, 1), False)
            self.input_format = "RGB"

        # ==========================================
        # 🟢 3. 使用官方函数加载完整的 LoftUP 模块并冻结
        # ==========================================
        loftup_weight_path = "/data/xyjiang/zlni/zlni/data/loftup/loftup_dinov2b.ckpt"
        self.loftup_refiner = load_loftup_checkpoint(
            upsampler_path=loftup_weight_path, 
            n_dim=768, 
            lr_pe_type="sine", 
            freeze=True  # 冻结参数以保护特征空间
        )

        # ==========================================
        # 🟢 4. 维度翻译器 (1024 <-> 768)
        # 因为 x1 和 x2 都是 1024 通道，我们可以直接复用这套投影层
        # ==========================================
        self.loftup_proj_in = nn.Conv2d(1024, 768, kernel_size=1)
        self.loftup_proj_out = nn.Conv2d(768, 1024, kernel_size=1)
        
        # 初始化投影器
        nn.init.dirac_(self.loftup_proj_in.weight[:768, :768])
        if self.loftup_proj_in.bias is not None: nn.init.zeros_(self.loftup_proj_in.bias)
        nn.init.dirac_(self.loftup_proj_out.weight[:768, :768])
        if self.loftup_proj_out.bias is not None: nn.init.zeros_(self.loftup_proj_out.bias)

    # 🟢 5. 重写 train 方法，确保 DVIS 训练时 LoftUP 保持在 eval 评估模式下
    def train(self, mode=True):
        super().train(mode)
        self.loftup_refiner.eval()

    def forward(self, x):
        """
        重写 forward 函数。
        使用 LoftUP 分别对 x1 (4倍) 和 x2 (2倍) 进行高清晰度引导上采样。
        """
        x_orig = x.clone()  # 备份一份输入图像作为指导图
        
        # =======================================================
        # 🟢 6. LoftUP 专属防御性数据清洗（与 AnyUp 严格保持对齐）
        # =======================================================
        # A. 逆向工程：反向恢复原始 [0, 255] 像素
        x_raw = x_orig * self.pixel_std + self.pixel_mean
        
        # B. 通道反转：如果 DVIS 输入的是 BGR 格式，自动转换为 LoftUP 要求的 RGB 格式
        if self.input_format == "BGR":
            x_raw = x_raw[:, [2, 1, 0], :, :]
            
        # C. 重新标准化：缩放到 [0, 1]，并使用标准的 ImageNet 均值和方差重新进行标准化
        anyup_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        anyup_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        x_for_loftup = (x_raw / 255.0 - anyup_mean) / anyup_std
        # =======================================================

        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
                    
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
                
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, finetune=finetune)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            
            # 计算 x1 和 x2 对应的上采样目标分辨率
            size_c1 = (c1.shape[-2], c1.shape[-1])  # 对应 1/4 尺寸 (4x上采样)
            size_c2 = (c2.shape[-2], c2.shape[-1])  # 对应 1/8 尺寸 (2x上采样)
            
            # 缩放防御性对齐后的引导图
            img_guide_c1 = F.interpolate(x_for_loftup, size=size_c1, mode='bilinear', align_corners=False)
            img_guide_c2 = F.interpolate(x_for_loftup, size=size_c2, mode='bilinear', align_corners=False)
            
            # 🟢 7. 使用 LoftUP 在无梯度环境下对 x1 进行直接上采样 (1024 -> 768 -> LoftUP -> 1024)
            with torch.no_grad():
                x1_768 = self.loftup_proj_in(x1)
                x1_refined_768 = self.loftup_refiner(lr_feats=x1_768, img=img_guide_c1)
                x1_refined = self.loftup_proj_out(x1_refined_768)
            
            # 🟢 8. 使用 LoftUP 在无梯度环境下对 x2 进行直接上采样 (1024 -> 768 -> LoftUP -> 1024)
            with torch.no_grad():
                x2_768 = self.loftup_proj_in(x2)
                x2_refined_768 = self.loftup_refiner(lr_feats=x2_768, img=img_guide_c2)
                x2_refined = self.loftup_proj_out(x2_refined_768)
            
            # x4 保持原生的双线性下采样
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # 融合增强特征
            c1, c2, c3, c4 = c1 + x1_refined, c2 + x2_refined, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONE_REGISTRY.register()
class D2VitAdapterDinoV2_LoftUP(DinoV2ViTAdapter_LoftUP, Backbone):
    """
    Detectron2 Wrapper for our custom LoftUP Adapter.
    """
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        # 🟢 将完整的 cfg 注入参数，使其传递给父类的 __init__ 拦截接收
        adapter_args["cfg"] = cfg
        super().__init__(**adapter_args)

        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x):
        assert x.dim() == 4, f"Takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


from research.modeling.anyup_utils.anyup.model import AnyUp

class DinoV2ViTAdapter_AnyUP(DinoV2ViTAdapter):
    def __init__(self, *args, **kwargs):
        # 🟢 在调用 super().__init__ 之前拦截 cfg 变量，防止父类抛出 TypeError
        cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)

        # ==========================================
        # 🟢 1. 注册 DVIS 的均值/方差以及通道格式
        # ==========================================
        if cfg is not None:
            # 注册为 buffer，会自动随模型转移到相同的 GPU 设备上
            self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1), False)
            self.input_format = cfg.INPUT.FORMAT
        else:
            # 兜底默认 ImageNet 255级配置
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.280, 103.530]).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.120, 57.375]).view(1, -1, 1, 1), False)
            self.input_format = "RGB"

        # ==========================================
        # 🟢 2. 初始化 AnyUp 模块
        # ==========================================
        self.anyup_refiner = AnyUp(
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False
        )

        # ==========================================
        # 🟢 3. 加载你在鱼类数据集上微调好的 10万步 AnyUp 权重
        # ==========================================
        anyup_ckpt_path = "/data/xyjiang/zlni/zlni/data/anyup/train/output/2026-07-26/02-36-24/version_0/model_100000steps.pth"
        if os.path.exists(anyup_ckpt_path):
            checkpoint = torch.load(anyup_ckpt_path, map_location="cpu")
            state_dict = checkpoint

            if isinstance(state_dict, dict) and 'anyup' in state_dict:
                state_dict = state_dict['anyup']
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            keys = list(state_dict.keys())
            for k in keys:
                if k.endswith("norm_q.weight"):
                    state_dict[k.replace("norm_q.weight", "norm_q.scale")] = state_dict.pop(k)
                elif k.endswith("norm_k.weight"):
                    state_dict[k.replace("norm_k.weight", "norm_k.scale")] = state_dict.pop(k)

            # 加载参数
            self.anyup_refiner.load_state_dict(state_dict) # 在 sam2_anyup.py 中对应的行是 self.anyup.load_state_dict(state_dict)
            print(f"🟢 DinoV2ViTAdapter_AnyUP: Loaded AnyUp checkpoint from {anyup_ckpt_path}")
        else:
            print(f"⚠️ Warning: DinoV2ViTAdapter_AnyUP checkpoint not found at {anyup_ckpt_path}")

        # ==========================================
        # 🟢 4. 严格冻结 AnyUp 参数
        # ==========================================
        for param in self.anyup_refiner.parameters():
            param.requires_grad = False
        self.anyup_refiner.eval()

    # 🟢 5. 重写 train 方法，确保 DVIS 训练时 AnyUp 处于 eval 评估模式下
    def train(self, mode=True):
        super().train(mode)
        self.anyup_refiner.eval()

    def forward(self, x):
        """
        重写 forward 函数。
        使用 AnyUp 分别对 x1 (4倍) 和 x2 (2倍) 进行高清晰度引导上采样。
        """
        x_orig = x.clone()  # 备份一份输入图像作为指导图
        
        # =======================================================
        # 🟢 6. AnyUp 专属防御性数据清洗
        # =======================================================
        # A. 逆向工程：反向乘以 Std 并加上 Mean，还原回原始 [0, 255] 像素
        x_raw = x_orig * self.pixel_std + self.pixel_mean
        
        # B. 通道反转：如果 DVIS 输入的是 BGR 格式，自动转换为 AnyUp 要求的 RGB 格式
        if self.input_format == "BGR":
            x_raw = x_raw[:, [2, 1, 0], :, :]
            
        # C. 重新标准化：缩放到 [0, 1]，并使用标准的 ImageNet 均值和方差重新进行标准化
        anyup_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        anyup_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        x_for_anyup = (x_raw / 255.0 - anyup_mean) / anyup_std
        # =======================================================

        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
                    
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
                
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, finetune=finetune)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            
            # 计算 x1 和 x2 对应的上采样目标分辨率
            size_c1 = (c1.shape[-2], c1.shape[-1])  # 对应 1/4 尺寸 (4x上采样)
            size_c2 = (c2.shape[-2], c2.shape[-1])  # 对应 1/8 尺寸 (2x上采样)
            
            # 使用经过防御性清洗的图片分别缩放到对应的目标分辨率
            img_guide_c1 = F.interpolate(x_for_anyup, size=size_c1, mode='bilinear', align_corners=False)
            img_guide_c2 = F.interpolate(x_for_anyup, size=size_c2, mode='bilinear', align_corners=False)
            
            # 在无梯度环境下对 x1 和 x2 进行高清晰引导上采样
            with torch.no_grad():
                x1_refined = self.anyup_refiner(img_guide_c1, x1)
                x2_refined = self.anyup_refiner(img_guide_c2, x2)
            
            # x4 需要缩小到 1/32（下采样），保持原生的双线性插值
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # 融合增强特征
            c1, c2, c3, c4 = c1 + x1_refined, c2 + x2_refined, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONE_REGISTRY.register()
class D2VitAdapterDinoV2_AnyUP(DinoV2ViTAdapter_AnyUP, Backbone):
    """
    Detectron2 Wrapper for our custom AnyUp Adapter.
    """
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        # 将 cfg 注入参数字典
        adapter_args["cfg"] = cfg
        super().__init__(**adapter_args)

        # 🌟 核心防御性补丁：防止全参解冻微调时 AttributeError 崩溃
        if not hasattr(self, "finetuned_interaction_indexes"):
            self.finetuned_interaction_indexes = list(range(len(self.interaction_indexes)))

        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x):
        assert x.dim() == 4, f"Takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

# =====================================================================
# 🟢 1. 全新交互块：在 DINOv2 blocks 输出后、Extractor 交叉注意力前进行实时细化
# =====================================================================
class InteractionBlockWithCls_AnyUpRefined(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        # 自定义可变形注意力特征提取器
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W, finetune=False,
                anyup_refiner=None, img_guide=None):
        # A. 运行冻结的 DINOv2 块 
        if finetune:
            x = torch.cat((cls, x), dim=1)
            for idx, blk in enumerate(blocks):
                x = blk(x)
            cls, x = x[:, :1, ], x[:, 1:, ]
        else:
            with torch.no_grad():
                x = torch.cat((cls, x), dim=1)
                for idx, blk in enumerate(blocks):
                    x = blk(x)
                cls, x = x[:, :1, ], x[:, 1:, ]

        # 🟢 B. 刚从 vit.block 出来，立即使用 AnyUp 进行等尺寸空间特征边缘锐化细化！
        if anyup_refiner is not None and img_guide is not None:
            bs, n, dim = x.shape
            # 转为 2D 特征图以满足 AnyUp 空间注意力格式要求
            x_2d = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            
            # 使用 AnyUp 精细化（在无梯度环境下运行，极其省显存）
            with torch.no_grad():
                x_2d_refined = anyup_refiner(img_guide, x_2d)
                
            # 重新压回 1D Token 序列
            x = x_2d_refined.flatten(2).transpose(1, 2).contiguous()

        # C. 此时，送入 Extractor 的特征 x 已经是被 AnyUp 边界对齐过的超清语义特征了！
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls


# =====================================================================
# 🟢 2. 全新适配器类：支持 Pre-Interact（交互前细化）模式
# =====================================================================
class DinoV2ViTAdapter_AnyUP_PreInteract(DinoV2ViTAdapter):
    def __init__(self, *args, **kwargs):
        # 拦截并获取 cfg，然后安全调用父类初始化
        cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)

        # ==========================================
        # 注册 DVIS 的均值/方差以及通道格式
        # ==========================================
        if cfg is not None:
            self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1), False)
            self.input_format = cfg.INPUT.FORMAT
        else:
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.280, 103.530]).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.120, 57.375]).view(1, -1, 1, 1), False)
            self.input_format = "RGB"

        # ==========================================
        # 初始化 AnyUp 模块
        # ==========================================
        self.anyup_refiner = AnyUp(
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False
        )

        # ==========================================
        # 加载你在鱼类数据集上微调好的 10万步 AnyUp 权重并做跨版本对齐
        # ==========================================
        anyup_ckpt_path = "/data/xyjiang/zlni/zlni/data/anyup/train/output/2026-07-26/02-36-24/version_0/model_100000steps.pth"
        if os.path.exists(anyup_ckpt_path):
            checkpoint = torch.load(anyup_ckpt_path, map_location="cpu")
            state_dict = checkpoint

            if isinstance(state_dict, dict) and 'anyup' in state_dict:
                state_dict = state_dict['anyup']
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            keys = list(state_dict.keys())
            for k in keys:
                if k.endswith("norm_q.weight"):
                    state_dict[k.replace("norm_q.weight", "norm_q.scale")] = state_dict.pop(k)
                elif k.endswith("norm_k.weight"):
                    state_dict[k.replace("norm_k.weight", "norm_k.scale")] = state_dict.pop(k)

            self.anyup_refiner.load_state_dict(state_dict)
            print(f"🟢 DinoV2ViTAdapter_AnyUP_PreInteract: Loaded AnyUp checkpoint from {anyup_ckpt_path}")
        else:
            print(f"⚠️ Warning: DinoV2ViTAdapter_AnyUP_PreInteract checkpoint not found at {anyup_ckpt_path}")

        # 严格冻结 AnyUp 参数
        for param in self.anyup_refiner.parameters():
            param.requires_grad = False
        self.anyup_refiner.eval()

        # ==========================================
        # 🟢 重建交互层：安全、动态地将原装交互块替换为我们的 AnyUp 细化交互块
        # ==========================================
        self.rebuild_interactions_with_anyup()

    def rebuild_interactions_with_anyup(self):
        """
        动态读取原有的参数配置，无损拷贝已经初始化好的可变形卷积参数，
        从而将原装的 InteractionBlock 升级为我们的 InteractionBlockWithCls_AnyUpRefined
        """
        new_interactions = []
        for i, old_layer in enumerate(self.interactions):
            dim = old_layer.extractor.attn.d_model
            num_heads = old_layer.extractor.attn.n_heads
            n_points = old_layer.extractor.attn.n_points
            
            # 实例化我们全新的交叉前细化交互层
            new_layer = InteractionBlockWithCls_AnyUpRefined(
                dim=dim,
                num_heads=num_heads,
                n_points=n_points,
                extra_extractor=(old_layer.extra_extractors is not None)
            )
            # 完整拷入初始化好的可变形注意力参数
            new_layer.extractor.load_state_dict(old_layer.extractor.state_dict())
            if old_layer.extra_extractors is not None:
                new_layer.extra_extractors.load_state_dict(old_layer.extra_extractors.state_dict())
                
            new_interactions.append(new_layer)
            
        self.interactions = nn.Sequential(*new_interactions)
        print(f"🟢 DinoV2ViTAdapter_AnyUP_PreInteract: Successfully rebuilt {len(self.interactions)} interaction layers with AnyUp!")

    def forward(self, x):
        """
        前置细化 forward 通路
        """
        x_orig = x.clone()  # 备份原图
        
        # AnyUp 防御性对齐
        dvis_mean = torch.tensor([123.675, 116.280, 103.530], device=x.device).view(1, 3, 1, 1)
        dvis_std  = torch.tensor([58.395, 57.120, 57.375], device=x.device).view(1, 3, 1, 1)
        x_raw = x_orig * dvis_std + dvis_mean
        if self.input_format == "BGR":
            x_raw = x_raw[:, [2, 1, 0], :, :]
        anyup_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        anyup_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_for_anyup = (x_raw / 255.0 - anyup_mean) / anyup_std

        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
                    
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # 🟢 准备 DINOv2 原生分辨率 (H, W) 的 guidance image
        img_guide = F.interpolate(x_for_anyup, size=(H, W), mode='bilinear', align_corners=False)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            
            # 🟢 在前向传播中，将 anyup_refiner 传入每一个交互模块内部运行
            x, c, cls = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, finetune=finetune,
                              anyup_refiner=self.anyup_refiner, img_guide=img_guide)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        # 在末端特征合并阶段，仅做原生的插值融合，因为特征在此前每一阶段已经通过 AnyUp 完成了高清边缘纠偏
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


# =====================================================================
# 🟢 3. 注册到 Detectron2 注册表的包装类
# =====================================================================
@BACKBONE_REGISTRY.register()
class D2VitAdapterDinoV2_AnyUP_PreInteract(DinoV2ViTAdapter_AnyUP_PreInteract, Backbone):
    """
    Detectron2 Wrapper for our custom Pre-Interact AnyUp Adapter.
    """
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        # 将 cfg 注入参数字典并完成初始化
        adapter_args["cfg"] = cfg
        super().__init__(**adapter_args)

        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x):
        assert x.dim() == 4, f"Takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32



# =====================================================================
# 🟢 1. 渐进式分辨率交互块（优化版：锁定最大 Stride 8，保护 Stage 3 原生 Stride 16 运行）
# =====================================================================
class InteractionBlockWithCls_AnyUP_Progressive(nn.Module):
    def __init__(self, stage_idx, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()
        self.stage_idx = stage_idx  # 记录当前阶段

        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W, finetune=False,
                anyup_refiner=None, img_guide_full=None):
        
        # A. 运行冻结的 DINOv2 块 
        if finetune:
            x = torch.cat((cls, x), dim=1)
            for idx, blk in enumerate(blocks):
                x = blk(x)
            cls, x = x[:, :1, ], x[:, 1:, ]
        else:
            with torch.no_grad():
                x = torch.cat((cls, x), dim=1)
                for idx, blk in enumerate(blocks):
                    x = blk(x)
                cls, x = x[:, :1, ], x[:, 1:, ]

        bs, n, dim = x.shape
        
        # B. 根据新设定的 Stride 8 渐进式逻辑，精确计算当前 x 的输入与输出尺寸
        if self.stage_idx == 0:
            H_in, W_in = H, W                  # 初始为 Stride 16
            target_H, target_W = H * 2, W * 2  # 出 Block 后，用 AnyUp 放大 2x 至 Stride 8
        elif self.stage_idx == 1:
            H_in, W_in = H * 2, W * 2          # 初始为 Stride 8
            target_H, target_W = H * 2, W * 2  # 出 Block 后，维持在 Stride 8
        elif self.stage_idx == 2:
            H_in, W_in = H * 2, W * 2          # 初始为 Stride 8
            target_H, target_W = H, W          # 出 Block 后，恢复到原生的 Stride 16
        elif self.stage_idx == 3:
            H_in, W_in = H, W                  # 🟢 完美听从建议：Stage 3 初始为 Stride 16
            target_H, target_W = H, W          # 🟢 完美听从建议：Stage 3 整个交互期间全部锁死在 Stride 16

        # 转为 2D 尺寸
        x_2d = x.transpose(1, 2).view(bs, dim, H_in, W_in).contiguous()

        # 🟢 C. 多阶段渐进式尺度转换
        if self.stage_idx == 0:
            # Stride 16 -> Stride 8：使用 AnyUp 进行 2x 高清引导上采样
            if anyup_refiner is not None and img_guide_full is not None:
                img_guide = F.interpolate(img_guide_full, size=(target_H, target_W), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    x_2d_refined = anyup_refiner(img_guide, x_2d)
            else:
                x_2d_refined = F.interpolate(x_2d, size=(target_H, target_W), mode='bilinear', align_corners=False)
        elif self.stage_idx == 1:
            # Stride 8 -> Stride 8 (维持尺寸，起到等尺寸特征细化作用)
            x_2d_refined = x_2d
        elif self.stage_idx == 2:
            # Stride 8 -> Stride 16 (下采样，退回到原生 Stride 16 空间)
            x_2d_refined = F.interpolate(x_2d, size=(target_H, target_W), mode='bilinear', align_corners=False)
        elif self.stage_idx == 3:
            # Stride 16 -> Stride 16 (🟢 完全无损，不进行任何下采样)
            x_2d_refined = x_2d

        # 重构回 1D Token 序列
        x = x_2d_refined.flatten(2).transpose(1, 2).contiguous()

        # 🟢 D. 动态构建特征抽取器所需的特征高宽参数
        curr_spatial_shapes = torch.as_tensor([(target_H, target_W)], dtype=torch.long, device=x.device)
        curr_level_start_index = curr_spatial_shapes.new_zeros((1,))

        # 运行特征抽取器，更新卷积特征 c
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=curr_spatial_shapes,
                           level_start_index=curr_level_start_index, H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=curr_spatial_shapes,
                              level_start_index=curr_level_start_index, H=H, W=W)
        return x, c, cls


# =====================================================================
# 🟢 2. 渐进式骨干适配器类：DinoV2ViTAdapter_AnyUP_Progressive
# =====================================================================
class DinoV2ViTAdapter_AnyUP_Progressive(DinoV2ViTAdapter):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)

        # 注册 DVIS 的均值/方差
        if cfg is not None:
            self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1), False)
            self.input_format = cfg.INPUT.FORMAT
        else:
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.280, 103.530]).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.120, 57.375]).view(1, -1, 1, 1), False)
            self.input_format = "RGB"

        # 初始化 AnyUp 模块
        self.anyup_refiner = AnyUp(
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False
        )

        # 加载 10万步权重并进行 RMSNorm 对齐
        anyup_ckpt_path = "/data/xyjiang/zlni/zlni/data/anyup/train/output/2026-07-26/02-36-24/version_0/model_100000steps.pth"
        if os.path.exists(anyup_ckpt_path):
            checkpoint = torch.load(anyup_ckpt_path, map_location="cpu")
            state_dict = checkpoint

            if isinstance(state_dict, dict) and 'anyup' in state_dict:
                state_dict = state_dict['anyup']
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            keys = list(state_dict.keys())
            for k in keys:
                if k.endswith("norm_q.weight"):
                    state_dict[k.replace("norm_q.weight", "norm_q.scale")] = state_dict.pop(k)
                elif k.endswith("norm_k.weight"):
                    state_dict[k.replace("norm_k.weight", "norm_k.scale")] = state_dict.pop(k)

            self.anyup_refiner.load_state_dict(state_dict)
            print(f"🟢 DinoV2ViTAdapter_AnyUP_Progressive: Loaded AnyUp checkpoint from {anyup_ckpt_path}")
        else:
            print(f"⚠️ Warning: DinoV2ViTAdapter_AnyUP_Progressive checkpoint not found at {anyup_ckpt_path}")

        # 严格冻结 AnyUp 参数
        for param in self.anyup_refiner.parameters():
            param.requires_grad = False
        self.anyup_refiner.eval()

        # 🟢 重塑接力交互层
        self.rebuild_interactions_with_progressive_anyup()

    def rebuild_interactions_with_progressive_anyup(self):
        new_interactions = []
        for i, old_layer in enumerate(self.interactions):
            dim = old_layer.extractor.attn.d_model
            num_heads = old_layer.extractor.attn.n_heads
            n_points = old_layer.extractor.attn.n_points
            
            new_layer = InteractionBlockWithCls_AnyUP_Progressive(
                stage_idx=i,  # 注入阶段索引
                dim=dim,
                num_heads=num_heads,
                n_points=n_points,
                extra_extractor=(old_layer.extra_extractors is not None)
            )
            # 拷贝权重
            new_layer.extractor.load_state_dict(old_layer.extractor.state_dict())
            if old_layer.extra_extractors is not None:
                new_layer.extra_extractors.load_state_dict(old_layer.extra_extractors.state_dict())
                
            new_interactions.append(new_layer)
            
        self.interactions = nn.Sequential(*new_interactions)
        print(f"🟢 DinoV2ViTAdapter_AnyUP_Progressive: Successfully rebuilt {len(self.interactions)} progressive layers!")

    def forward(self, x):
        x_orig = x.clone()  # 备份原图
        
        # AnyUp 防御性对齐
        dvis_mean = torch.tensor([123.675, 116.280, 103.530], device=x.device).view(1, 3, 1, 1)
        dvis_std  = torch.tensor([58.395, 57.120, 57.375], device=x.device).view(1, 3, 1, 1)
        x_raw = x_orig * dvis_std + dvis_mean
        if self.input_format == "BGR":
            x_raw = x_raw[:, [2, 1, 0], :, :]
        anyup_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        anyup_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_for_anyup = (x_raw / 255.0 - anyup_mean) / anyup_std

        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
                    
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # 准备原始无缩放的引导图
        img_guide_full = x_for_anyup

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            
            # 传入全尺寸引导图
            x, c, cls = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, finetune=finetune,
                              anyup_refiner=self.anyup_refiner, img_guide_full=img_guide_full)
            
            # 按照各阶段退出时的真实特征图高宽进行 Reshape 记录
            if i == 0:
                curr_H, curr_W = H * 2, W * 2  # Stride 8
            elif i == 1:
                curr_H, curr_W = H * 2, W * 2  # Stride 8
            elif i == 2:
                curr_H, curr_W = H, W          # Stride 16
            elif i == 3:
                curr_H, curr_W = H, W          # Stride 16 (🟢 完全对齐你的方案)
                
            outs.append(x.transpose(1, 2).view(bs, dim, curr_H, curr_W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        # 🟢 在最末端进行通道融合与最终的插值对齐
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            
            # x1 (Stride 8) -> 放大 2 倍到 Stride 4
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
            # x2 (Stride 8) -> 无需任何变化，直接融合到 c2(Stride 8)
            # x3 (Stride 16) -> 无需任何变化，直接融合到 c3(Stride 16)
            # x4 (Stride 16) -> 🟢 完美实现你的建议：DINOv2 在 Stride 16 下处理完后，在最后一刻下采样到 Stride 32
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


# =====================================================================
# 🟢 3. 注册包装类
# =====================================================================
@BACKBONE_REGISTRY.register()
class D2VitAdapterDinoV2_AnyUP_Progressive(DinoV2ViTAdapter_AnyUP_Progressive, Backbone):
    """
    Detectron2 Wrapper for our custom Progressive AnyUp Adapter.
    """
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        # 将 cfg 注入参数字典并完成初始化
        adapter_args["cfg"] = cfg
        super().__init__(**adapter_args)

        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x):
        assert x.dim() == 4, f"Takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


# =====================================================================
# 🟢 1. 终极双通路直接上采样交互块：每层独立一步到位重构，不污染下级 blocks
# =====================================================================
class InteractionBlockWithCls_DualPath_AnyUp_Direct(nn.Module):
    def __init__(self, stage_idx, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()
        self.stage_idx = stage_idx  # 标识当前是第几个接力阶段

        # 自定义可变形注意力特征提取器
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W, finetune=False,
                anyup_refiner=None, img_guide_full=None):
        # A. 运行当前阶段冻结的 DINOv2 块 (输入 x 始终为原装的 Stride 16)
        if finetune:
            x = torch.cat((cls, x), dim=1)
            for idx, blk in enumerate(blocks):
                x = blk(x)
            cls, x = x[:, :1, ], x[:, 1:, ]
        else:
            with torch.no_grad():
                x = torch.cat((cls, x), dim=1)
                for idx, blk in enumerate(blocks):
                    x = blk(x)
                cls, x = x[:, :1, ], x[:, 1:, ]

        # 🟢 B.【双通路核心设计 1】
        # x_for_next_stage: 必须强行克隆并保持原生 Stride 16 (H, W)，用于安全输送给下一个 ViT 块
        x_for_next_stage = x.clone()

        # 🟢 C. 计算当前阶段出 Block 后的【一步到位目标分辨率】：
        if self.stage_idx == 0:
            target_H, target_W = H * 4, W * 4  # Stage 0 直接放大 4 倍至 Stride 4 (res2尺寸)
        elif self.stage_idx == 1:
            target_H, target_W = H * 2, W * 2  # Stage 1 直接放大 2 倍至 Stride 8 (res3尺寸)
        elif self.stage_idx == 2:
            target_H, target_W = H, W          # Stage 2 保持 Stride 16 进行等尺寸特征对齐细化 (res4尺寸)
        elif self.stage_idx == 3:
            target_H, target_W = H // 2, W // 2  # Stage 3 下采样至 Stride 32 (res5尺寸)

        # 2D Reshape（输入的 x 一定是 Stride 16 尺寸 H x W）
        bs, n, dim = x.shape
        x_2d = x.transpose(1, 2).view(bs, dim, H, W).contiguous()

        # 🟢 D. 使用 AnyUp 直接一步到位重构出目标分辨率（无中间插值）
        if self.stage_idx == 0:
            # Stride 16 -> Stride 4 (4x AnyUp 引导上采样)
            if anyup_refiner is not None and img_guide_full is not None:
                img_guide = F.interpolate(img_guide_full, size=(target_H, target_W), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    x_2d_refined = anyup_refiner(img_guide, x_2d)
            else:
                x_2d_refined = F.interpolate(x_2d, size=(target_H, target_W), mode='bilinear', align_corners=False)
        elif self.stage_idx == 1:
            # Stride 16 -> Stride 8 (2x AnyUp 引导上采样)
            if anyup_refiner is not None and img_guide_full is not None:
                img_guide = F.interpolate(img_guide_full, size=(target_H, target_W), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    x_2d_refined = anyup_refiner(img_guide, x_2d)
            else:
                x_2d_refined = F.interpolate(x_2d, size=(target_H, target_W), mode='bilinear', align_corners=False)
        elif self.stage_idx == 2:
            # Stride 16 -> Stride 16 (1x AnyUp 引导等尺寸精细化)
            if anyup_refiner is not None and img_guide_full is not None:
                img_guide = F.interpolate(img_guide_full, size=(target_H, target_W), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    x_2d_refined = anyup_refiner(img_guide, x_2d)
            else:
                x_2d_refined = x_2d
        elif self.stage_idx == 3:
            # Stride 16 -> Stride 32 (F.interpolate 下采样)
            x_2d_refined = F.interpolate(x_2d, size=(target_H, target_W), mode='bilinear', align_corners=False)

        # 重构回 1D Token 序列以通过特征抽取器
        x_refined_flat = x_2d_refined.flatten(2).transpose(1, 2).contiguous()

        # 🟢 E. 动态计算当前分辨率下特征抽取器（MSDeformAttn）所需的空间参数
        curr_spatial_shapes = torch.as_tensor([(target_H, target_W)], dtype=torch.long, device=x.device)
        curr_level_start_index = curr_spatial_shapes.new_zeros((1,))

        # 🟢 F. 用经过 AnyUp 直接放大的、高精度边界特征 x_refined_flat 去刷新卷积先验 c
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x_refined_flat, spatial_shapes=curr_spatial_shapes,
                           level_start_index=curr_level_start_index, H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x_refined_flat, spatial_shapes=curr_spatial_shapes,
                              level_start_index=curr_level_start_index, H=H, W=W)
                
        # 返回:
        # 1. x_for_next_stage: 保证 DINOv2 下一阶段依然能接收到原生 Stride 16 的 Token 数量
        # 2. c: 得到极高质量强化的卷积先验特征
        # 3. cls: 类别 Token
        # 4. x_2d_refined: 🟢 直接作为 2D 目标分辨率特征返回，在最后直接和 c 相加！
        return x_for_next_stage, c, cls, x_2d_refined


# =====================================================================
# 🟢 2. 双通路骨干适配器类：DinoV2ViTAdapter_AnyUP_DualPath_Direct
# =====================================================================
class DinoV2ViTAdapter_AnyUP_DualPath_Direct(DinoV2ViTAdapter):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)

        # 注册 DVIS 的均值/方差以及通道格式
        if cfg is not None:
            self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1), False)
            self.input_format = cfg.INPUT.FORMAT
        else:
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.280, 103.530]).view(1, -1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.120, 57.375]).view(1, -1, 1, 1), False)
            self.input_format = "RGB"

        # 初始化 AnyUp 模块
        self.anyup_refiner = AnyUp(
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False
        )

        # 加载 10万步权重并进行 RMSNorm 对齐
        anyup_ckpt_path = "/data/xyjiang/zlni/zlni/data/anyup/train/output/2026-07-26/02-36-24/version_0/model_100000steps.pth"
        if os.path.exists(anyup_ckpt_path):
            checkpoint = torch.load(anyup_ckpt_path, map_location="cpu")
            state_dict = checkpoint

            if isinstance(state_dict, dict) and 'anyup' in state_dict:
                state_dict = state_dict['anyup']
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            keys = list(state_dict.keys())
            for k in keys:
                if k.endswith("norm_q.weight"):
                    state_dict[k.replace("norm_q.weight", "norm_q.scale")] = state_dict.pop(k)
                elif k.endswith("norm_k.weight"):
                    state_dict[k.replace("norm_k.weight", "norm_k.scale")] = state_dict.pop(k)

            self.anyup_refiner.load_state_dict(state_dict)
            print(f"🟢 DinoV2ViTAdapter_AnyUP_DualPath_Direct: Loaded AnyUp checkpoint from {anyup_ckpt_path}")
        else:
            print(f"⚠️ Warning: DinoV2ViTAdapter_AnyUP_DualPath_Direct checkpoint not found at {anyup_ckpt_path}")

        # 严格冻结 AnyUp 参数
        for param in self.anyup_refiner.parameters():
            param.requires_grad = False
        self.anyup_refiner.eval()

        # 🟢 重构接力交互层
        self.rebuild_interactions_with_dualpath_anyup_direct()

    def rebuild_interactions_with_dualpath_anyup_direct(self):
        new_interactions = []
        for i, old_layer in enumerate(self.interactions):
            dim = old_layer.extractor.attn.d_model
            num_heads = old_layer.extractor.attn.n_heads
            n_points = old_layer.extractor.attn.n_points
            
            # 实例化我们全新的双通路直接上采样交互模块
            new_layer = InteractionBlockWithCls_DualPath_AnyUp_Direct(
                stage_idx=i,  # 注入阶段索引
                dim=dim,
                num_heads=num_heads,
                n_points=n_points,
                extra_extractor=(old_layer.extra_extractors is not None)
            )
            # 拷贝已初始化好的可变形注意力参数
            new_layer.extractor.load_state_dict(old_layer.extractor.state_dict())
            if old_layer.extra_extractors is not None:
                new_layer.extra_extractors.load_state_dict(old_layer.extra_extractors.state_dict())
                
            new_interactions.append(new_layer)
            
        self.interactions = nn.Sequential(*new_interactions)
        print(f"🟢 DinoV2ViTAdapter_AnyUP_DualPath_Direct: Successfully rebuilt {len(self.interactions)} dual-path layers!")

    def forward(self, x):
        x_orig = x.clone()  # 备份原图
        
        # AnyUp 防御性对齐
        dvis_mean = torch.tensor([123.675, 116.280, 103.530], device=x.device).view(1, 3, 1, 1)
        dvis_std  = torch.tensor([58.395, 57.120, 57.375], device=x.device).view(1, 3, 1, 1)
        x_raw = x_orig * dvis_std + dvis_mean
        if self.input_format == "BGR":
            x_raw = x_raw[:, [2, 1, 0], :, :]
        anyup_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        anyup_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_for_anyup = (x_raw / 255.0 - anyup_mean) / anyup_std

        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
                    
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # 准备原始无缩放的引导图
        img_guide_full = x_for_anyup

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            
            # 🟢 调用双通路直接重构上采样交互
            # 返回标准的 x 传给下一轮循环，返回在内部已经由 AnyUp 一步到位重构好的 x_refined_2d
            x, c, cls, x_refined_2d = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                                            deform_inputs1, deform_inputs2, H, W, finetune=finetune,
                                            anyup_refiner=self.anyup_refiner, img_guide_full=img_guide_full)
            
            # 直接将重构好的多尺度 2D 特征图追加到 outs 中
            outs.append(x_refined_2d)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        # 🟢【终极融合】完全零插值！
        # 因为 x1, x2, x3, x4 在交互块内部就已经直接由 AnyUp 在最精准的几何边界下重构好了！
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


# =====================================================================
# 🟢 3. 注册包装类
# =====================================================================
@BACKBONE_REGISTRY.register()
class D2VitAdapterDinoV2_AnyUP_DualPath_Direct(DinoV2ViTAdapter_AnyUP_DualPath_Direct, Backbone):
    """
    Detectron2 Wrapper for our custom Dual-Path Direct AnyUp Adapter.
    """
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
        )
        # 将 cfg 注入参数字典并完成初始化
        adapter_args["cfg"] = cfg
        super().__init__(**adapter_args)

        # 🌟 核心防御性补丁：防止全参解冻微调时 AttributeError 崩溃
        if not hasattr(self, "finetuned_interaction_indexes"):
            self.finetuned_interaction_indexes = list(range(len(self.interaction_indexes)))

        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x):
        assert x.dim() == 4, f"Takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


class DepthDinoV2ViTAdapter(nn.Module):
    """
    基于 DinoV2ViTAdapter 的深度融合版本。
    在原结构基础上增加深度分支的 SpatialPriorModule，并在 c1-c4 上融合 RGB 与深度特征。
    通过 fusion_type 控制融合方式：'cmx', 'tokenfusion', 'cen'。
    """
    def __init__(
        self,
        vit_module=None,
        pretrain_size=224,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
        freeze_backbone=False,
        finetune=False,
        finetune_indexes=[0, ],
        fusion_type='none',
        depth_in_channels=1,
    ):
        super().__init__()

        self.num_block = len(vit_module.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = vit_module.embed_dim
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type

        self.drop_path_rate = vit_module.configs_dict['drop_path_rate']
        self.norm_layer = vit_module.norm_layer

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        # RGB 分支 SPM（输入 3 通道）
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim, in_channels=3, with_cp=False
        )

        # 深度分支 SPM（输入通道可配置）
        if fusion_type != 'none':
            self.depth_spm = SpatialPriorModule(
                inplanes=conv_inplane, embed_dim=embed_dim, in_channels=depth_in_channels, with_cp=False
            )
            # 为 c2~c4 创建融合层
            self.fusion_layers = nn.ModuleList([
                build_fusion_layer(fusion_type, embed_dim) for _ in range(2)
            ])
        else:
            self.depth_spm = None
            self.fusion_layers = None

        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls_Efficient(
                dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                init_values=init_values, drop_path=self.drop_path_rate,
                norm_layer=self.norm_layer, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                with_cp=with_cp
            )
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        if self.depth_spm is not None:
            self.depth_spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        if self.fusion_layers is not None:
            self.fusion_layers.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        self.vit_module = vit_module
        # 以下冻结/微调逻辑与原类完全一致
        if freeze_backbone:
            for p in self.vit_module.parameters():
                p.requires_grad_(False)
            if finetune:
                finetuned_interaction_indexes = finetune_indexes
                self.finetuned_interaction_indexes = finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    for p in self.vit_module.blocks[indexes[0]:indexes[-1] + 1].parameters():
                        p.requires_grad_(True)

        self.freeze_backbone = freeze_backbone
        self.finetune = finetune

        if not hasattr(self, "finetuned_interaction_indexes"):
            self.finetuned_interaction_indexes = list(range(len(self.interaction_indexes)))

        if not self.freeze_backbone:
            if hasattr(self.vit_module, "norm") and self.vit_module.norm is not None:
                for p in self.vit_module.norm.parameters():
                    p.requires_grad = False
            if hasattr(self.vit_module, "mask_token") and self.vit_module.mask_token is not None:
                self.vit_module.mask_token.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _token_to_img(self, x, H, W):
        """将 token 序列 (B, L, C) 转为 2D 特征图 (B, C, H, W)"""
        B, L, C = x.shape
        return x.transpose(1, 2).view(B, C, H, W).contiguous()

    def _img_to_token(self, x):
        """将 2D 特征图 (B, C, H, W) 转为 token 序列 (B, HW, C)"""
        B, C, H, W = x.shape
        return x.view(B, C, -1).transpose(1, 2).contiguous()

    def forward(self, x, depth=None):
        """
        Args:
            x: RGB 输入 (B, 3, H, W)
            depth: 深度输入 (B, 1, H, W) 或 None
        """
        if self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune and self.training:
                finetuned_interaction_indexes = self.finetuned_interaction_indexes
                for idx in finetuned_interaction_indexes:
                    indexes = self.interaction_indexes[idx]
                    self.vit_module.blocks[indexes[0]:indexes[-1] + 1].train()
        deform_inputs1, deform_inputs2 = deform_inputs(x)


        # RGB SPM
        c1, c2, c3, c4 = self.spm(x)

        # 深度融合（如果启用）
        if depth is not None and self.fusion_type != 'none':
            d1, d2, d3, d4 = self.depth_spm(depth)

            # 各尺度空间尺寸（根据 c1 计算）
            h_c1, w_c1 = c1.shape[-2], c1.shape[-1]
            h_c2, w_c2 = h_c1 // 2, w_c1 // 2
            h_c3, w_c3 = h_c2 // 2, w_c2 // 2
            h_c4, w_c4 = h_c3 // 2, w_c3 // 2

            # 融合 c1（本身就是 2D）
            #c1 = self.fusion_layers[1](c1, d1)
            c1 = c1 + d1
            c2 = c2 + d2


            # 融合 c2, c3, c4（先转为 2D）
            # c2_img = self._token_to_img(c2, h_c2, w_c2)
            # d2_img = self._token_to_img(d2, h_c2, w_c2) if d2.dim() == 3 else d2
            # c2_fused_img = self.fusion_layers[0](c2_img, d2_img)
            # c2 = self._img_to_token(c2_fused_img)

            c3_img = self._token_to_img(c3, h_c3, w_c3)
            d3_img = self._token_to_img(d3, h_c3, w_c3) if d3.dim() == 3 else d3
            c3_fused_img = self.fusion_layers[0](c3_img, d3_img)
            c3 = self._img_to_token(c3_fused_img)

            c4_img = self._token_to_img(c4, h_c4, w_c4)
            d4_img = self._token_to_img(d4, h_c4, w_c4) if d4.dim() == 3 else d4
            c4_fused_img = self.fusion_layers[1](c4_img, d4_img)
            c4 = self._img_to_token(c4_fused_img)

        # 后续与原 forward 一致
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if self.finetune:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.finetune and i in self.finetuned_interaction_indexes:
                finetune = True
            else:
                finetune = False
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.vit_module.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H, W, finetune=finetune)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

@BACKBONE_REGISTRY.register()
class D2DepthVitAdapterDinoV2(DepthDinoV2ViTAdapter, Backbone):
    """
    Detectron2 注册版本，从 cfg 读取深度融合相关参数。
    """
    def __init__(self, cfg, input_shape):
        name = cfg.MODEL.VIT_ADAPTER.NAME
        backbone_weight = cfg.MODEL.VIT_ADAPTER.VIT_WEIGHT
        freeze_backbone = cfg.MODEL.VIT_ADAPTER.FREEZE_VIT
        finetune = cfg.MODEL.VIT_ADAPTER.FINETUNE
        finetune_indexes = cfg.MODEL.VIT_ADAPTER.FINETUNE_INDEXES
        with_cp = cfg.MODEL.VIT_ADAPTER.WITH_CP
        fusion_type = cfg.MODEL.VIT_ADAPTER.FUSION_TYPE if hasattr(cfg.MODEL.VIT_ADAPTER, 'FUSION_TYPE') else 'none'
        depth_in_channels = cfg.MODEL.VIT_ADAPTER.DEPTH_IN_CHANNELS if hasattr(cfg.MODEL.VIT_ADAPTER, 'DEPTH_IN_CHANNELS') else 1

        # 获取适配器参数（需要修改 get_adapter_args 支持 fusion_type 和 depth_in_channels）
        adapter_args = get_adapter_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            finetune=finetune,
            finetune_indexes=finetune_indexes,
            with_cp=with_cp,
            fusion_type=fusion_type,
            depth_in_channels=depth_in_channels,
        )
        super().__init__(**adapter_args)

        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.vit_module.embed_dim,
            "res3": self.vit_module.embed_dim,
            "res4": self.vit_module.embed_dim,
            "res5": self.vit_module.embed_dim,
        }

    def forward(self, x, depth=None):
        """
        Args:
            x: Tensor of shape (N, C, H, W). H, W must be a multiple of ``size_divisibility``.
            depth: Optional tensor of shape (N, 1, H, W) or None.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x, depth)   # 调用 DepthDinoV2ViTAdapter.forward
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.vit_module.embed_dim, stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32