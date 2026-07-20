import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY, build_resnet_backbone
from detectron2.layers import ShapeSpec
from einops import rearrange
import os

from research.modeling.jafar_utils import CrossAttentionBlock, ResBlock, SFTModulation, RoPE
from research.modeling.jafar_utils.coordinate import create_coordinate
from detectron2.checkpoint import DetectionCheckpointer

class JAFAR(nn.Module):
    def __init__(
        self,
        input_dim=3,          
        qk_dim=256,          
        v_dim=256,            
        feature_dim=None,
        kernel_size=1,
        num_heads=4,
        **kwargs,
    ):
        super().__init__()
        self.image_encoder = self._make_encoder(input_dim, qk_dim, kernel_size=3)
        self.key_encoder   = self._make_encoder(qk_dim, qk_dim, kernel_size=1)
        self.query_encoder = self._make_encoder(qk_dim, qk_dim, kernel_size=1)

        self.norm = nn.GroupNorm(num_groups=8, num_channels=qk_dim, affine=False)

        self.key_features_encoder = nn.Conv2d(v_dim, qk_dim, kernel_size=1, bias=False)
        self.cross_decode = CrossAttentionBlock(qk_dim, qk_dim, v_dim, num_heads)

        self.sft_key = SFTModulation(qk_dim, qk_dim)
        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def _make_encoder(self, in_dim, out_dim, kernel_size, num_layers=2):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size//2,
                      padding_mode="reflect", bias=False),
            *[ResBlock(out_dim, out_dim, kernel_size=1, num_groups=8,
                       pad_mode="reflect", norm_fn=nn.GroupNorm,
                       activation_fn=nn.SiLU, use_conv_shortcut=False)
              for _ in range(num_layers)]
        )

    def forward(self, image, features, output_size):
        encoded_image = self.image_encoder(image)
        B, C, H_i, W_i = encoded_image.shape

        # positional encoding ...
        coords = create_coordinate(H_i, W_i)
        encoded_image = rearrange(encoded_image, "b c h w -> b (h w) c")
        encoded_image = self.rope(encoded_image, coords)
        encoded_image = rearrange(encoded_image, "b (h w) c -> b c h w", h=H_i)

        queries = self.query_encoder(encoded_image)
        queries = F.adaptive_avg_pool2d(queries, output_size=(output_size[0]//2, output_size[1]//2))
        queries = self.norm(queries)

        keys = self.key_encoder(encoded_image)

        downsampled_feat_size = (output_size[0]//2, output_size[1]//2)
        keys = F.adaptive_avg_pool2d(keys, downsampled_feat_size)

        # 对特征图也降采样（Value）
        features_down = F.adaptive_avg_pool2d(features, downsampled_feat_size)
        feat_map = self.key_features_encoder(F.normalize(features_down, dim=1))
        keys = self.sft_key(keys, feat_map)

        # cross attention 在降采样后的尺寸上计算
        out = self.cross_decode(queries, keys, features_down)   # 返回 (B, H_out*W_out//4, v_dim)
        # 上采样回 output_size
        h_out, w_out = output_size
        out = rearrange(out, "b (h w) c -> b c h w", h=downsampled_feat_size[0], w=downsampled_feat_size[1])
        out = F.interpolate(out, size=(h_out, w_out), mode='bilinear', align_corners=False)

        return out


class JAFARResNetBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        base_resnet = build_resnet_backbone(cfg, input_shape)

        self.resnet = base_resnet
        base_shapes = base_resnet.output_shape()

        self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        self.lateral_convs = nn.ModuleDict()
        for name in self._out_features:
            in_ch = base_shapes[name].channels
            self.lateral_convs[name] = nn.Conv2d(in_ch, 256, kernel_size=1, bias=False)

        self.jafar = JAFAR(
            input_dim=3,       
            qk_dim=256,
            v_dim=256,
            num_heads=8,      
        )

        self._out_feature_channels = {name: 256 for name in self._out_features}
        self._out_feature_strides = {
            name: base_shapes[name].stride for name in self._out_features
        }
        weights_path = cfg.MODEL.WEIGHTS
        if weights_path and (os.path.isfile(weights_path) or weights_path.startswith("detectron2://")):
            print(f"[JAFAR] Manually intercepting and loading weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            
            # 1. 提取并修复 ResNet 的前缀
            resnet_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.res") or k.startswith("backbone.stem"):
                    new_k = k.replace("backbone.", "")
                    resnet_state_dict[new_k] = v
            
            # 加载ResNet
            missing, unexpected = self.resnet.load_state_dict(resnet_state_dict, strict=False)
            print(f"[JAFAR] -> ResNet weights loaded! Missing: {len(missing)} keys.")

            # 2. 提取并加载 JAFAR 的权重
            jafar_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.jafar."):
                    new_k = k.replace("backbone.jafar.", "")
                    jafar_state_dict[new_k] = v
            
            if jafar_state_dict:
                self.jafar.load_state_dict(jafar_state_dict, strict=False)
                print(f"[JAFAR] -> JAFAR weights loaded!")
            else:
                print(f"[JAFAR] -> No JAFAR weights found. Training JAFAR from scratch.")

    def forward(self, x):
        res_features = self.resnet(x)   # dict: name -> tensor

        outputs = {}
        # 只对 res3, res4, res5 应用 JAFAR，res2 直接降维后输出（跳过 JAFAR）
        apply_jafar_on = {"res3", "res4", "res5"}

        for name in self._out_features:
            feat = res_features[name]
            # 1x1 降维到 256
            feat = self.lateral_convs[name](feat)   # (B, 256, h, w)

            if name in apply_jafar_on:
                h, w = feat.shape[2:]
                # 可选：对 query/key 的特征图降采样以进一步节省显存（如 stride=2）
                # 在 JAFAR 内部或外部对 query 和 key 做自适应平均池化缩小尺寸
                enhanced = self.jafar(image=x, features=feat, output_size=(h, w))
                outputs[name] = enhanced
            else:
                # res2 不经过 JAFAR，直接输出
                outputs[name] = feat

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_resnet_jafar_backbone(cfg, input_shape: ShapeSpec):
    """
    返回 JAFARResNetBackbone 实例
    """
    return JAFARResNetBackbone(cfg, input_shape)