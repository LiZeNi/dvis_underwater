import os
import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

# 导入 AnyUp 核心模型结构
from research.modeling.anyup_utils.anyup.model import AnyUp
from ..sam2_utils.hieradet import Hiera


@BACKBONE_REGISTRY.register()
class SAM2AnyupBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        embed_dim = cfg.MODEL.SAM2_HIERA.EMBED_DIM
        num_heads = cfg.MODEL.SAM2_HIERA.NUM_HEADS
        stages = tuple(cfg.MODEL.SAM2_HIERA.STAGES)
        global_att_blocks = tuple(cfg.MODEL.SAM2_HIERA.GLOBAL_ATT_BLOCKS)
        window_spec = tuple(cfg.MODEL.SAM2_HIERA.WINDOW_SPEC)
        drop_path_rate = cfg.MODEL.SAM2_HIERA.DROP_PATH_RATE
        sam2_channels = cfg.MODEL.SAM2_HIERA.CHANNEL_LIST

        # 1. 初始化 SAM2 Hiera 主干网络
        self.hiera = Hiera(
            embed_dim=embed_dim,           
            num_heads=num_heads,            
            stages=stages,    
            global_att_blocks=global_att_blocks, 
            window_pos_embed_bkg_spatial_size=(7, 7), 
            window_spec=window_spec,  
            drop_path_rate=drop_path_rate
        )

        # 2. 初始化 AnyUp 模块
        self.anyup = AnyUp(
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False
        )

        # 3. 加载已经微调训练好的 AnyUp 权重
        anyup_ckpt_path = "/data/xyjiang/zlni/zlni/data/anyup/train/output/2026-07-26/02-36-24/version_0/model_100000steps.pth"
        if os.path.exists(anyup_ckpt_path):
            checkpoint = torch.load(anyup_ckpt_path, map_location="cpu")
            state_dict = checkpoint
            # 兼容全量 checkpoint 与 纯 state_dict 权重载入
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

            self.anyup.load_state_dict(state_dict) # 在 sam2_anyup.py 中对应的行是 self.anyup.load_state_dict(state_dict)
            print(f"🟢 SAM2HieraBackbone: Loaded AnyUp checkpoint from {anyup_ckpt_path}")
        else:
            print(f"⚠️ Warning: SAM2HieraBackbone: checkpoint not found at {anyup_ckpt_path}")

        # 4. 彻底冻结 AnyUp 参数（不计算梯度，不参与反向传播）
        for param in self.anyup.parameters():
            param.requires_grad = False
        self.anyup.eval()  # 设置初始评估模式

        # 5. 定义 Hiera 原始特征通道对齐的 1x1 卷积
        self.align_res2 = nn.Conv2d(sam2_channels[0], 256,  kernel_size=1)
        self.align_res3 = nn.Conv2d(sam2_channels[1], 512,  kernel_size=1)
        self.align_res4 = nn.Conv2d(sam2_channels[2], 1024, kernel_size=1)
        self.align_res5 = nn.Conv2d(sam2_channels[3], 2048, kernel_size=1)

        # 6. 定义 AnyUp 上采样后的通道对齐 1x1 卷积
        # 🟢 根据论文设计：所有的上采样特征均源自 stage5 (channels: sam2_channels[3])
        self.align_res4_anyup = nn.Conv2d(sam2_channels[3], 1024, kernel_size=1)  # 32 -> 16 尺度对齐
        self.align_res3_anyup = nn.Conv2d(sam2_channels[3], 512,  kernel_size=1)  # 32 -> 8  尺度对齐
        self.align_res2_anyup = nn.Conv2d(sam2_channels[3], 256,  kernel_size=1)  # 32 -> 4  尺度对齐
        
        # 初始化所有卷积参数
        conv_layers = [
            self.align_res2, self.align_res3, self.align_res4, self.align_res5,
            self.align_res2_anyup, self.align_res3_anyup, self.align_res4_anyup
        ]
        for m in conv_layers:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = {
            "res2": 256,
            "res3": 512,
            "res4": 1024,
            "res5": 2048,
        }
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

    # 7. 覆盖 train() 方法，确保 DVIS 训练时 AnyUp 强制保持在 eval 评估模式下
    def train(self, mode=True):
        super().train(mode)
        self.anyup.eval()

    def forward(self, x):
        # 提取 Hiera 主干的多尺度特征 [stage1, stage2, stage3, stage4]，strides 分别为 [4, 8, 16, 32]
        hiera_outputs = self.hiera(x)
        
        # 🟢 论文设定：以最深层、富含高级语义的特征 (stride 32) 作为唯一的低分辨率上采样源 p
        p = hiera_outputs[3]

        # 确定需要对齐的目标浅层分辨率尺寸
        size2 = hiera_outputs[0].shape[-2:]  # 目标尺寸 (H/4, W/4)
        size3 = hiera_outputs[1].shape[-2:]  # 目标尺寸 (H/8, W/8)
        size4 = hiera_outputs[2].shape[-2:]  # 目标尺寸 (H/16, W/16)

        # 🟢 在无梯度的独立上下文里运行 AnyUp 进行多尺度直接上采样
        with torch.no_grad():
            # 32 -> 16 直接上采样
            up_res5_to_16 = self.anyup(x, p, output_size=size4)
            # 32 -> 8 直接上采样
            up_res5_to_8  = self.anyup(x, p, output_size=size3)
            # 32 -> 4 直接上采样
            up_res5_to_4  = self.anyup(x, p, output_size=size2)

        features = {}
        
        # res5 (stride 32): 维持原有的语义输出并对齐通道
        features["res5"] = self.align_res5(p)  
        
        # res4 (stride 16): 融合原本的 res4 与由 AnyUp 直接从最底层语义上采上来的特征
        features["res4"] = self.align_res4(hiera_outputs[2]) + self.align_res4_anyup(up_res5_to_16)
        
        # res3 (stride 8): 融合原本的 res3 与由 AnyUp 直接从最底层语义上采上来的特征
        features["res3"] = self.align_res3(hiera_outputs[1]) + self.align_res3_anyup(up_res5_to_8)
        
        # res2 (stride 4): 融合原本的 res2 与由 AnyUp 直接从最底层语义上采上来的特征
        features["res2"] = self.align_res2(hiera_outputs[0]) + self.align_res2_anyup(up_res5_to_4)
        
        return features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }