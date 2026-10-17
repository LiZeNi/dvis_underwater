import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from detectron2.layers import ShapeSpec
from ..sam2_utils.hieradet import Hiera
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec
from ..loftup_utils.upsamplers.upsamplers import LoftUp, UpsamplerwithChannelNorm, ChannelNorm,load_loftup_checkpoint

@BACKBONE_REGISTRY.register()
class SAM2LoftupBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        embed_dim = cfg.MODEL.SAM2_HIERA.EMBED_DIM
        num_heads = cfg.MODEL.SAM2_HIERA.NUM_HEADS
        stages = tuple(cfg.MODEL.SAM2_HIERA.STAGES)
        global_att_blocks = tuple(cfg.MODEL.SAM2_HIERA.GLOBAL_ATT_BLOCKS)
        window_spec = tuple(cfg.MODEL.SAM2_HIERA.WINDOW_SPEC)
        drop_path_rate = cfg.MODEL.SAM2_HIERA.DROP_PATH_RATE
        sam2_channels = cfg.MODEL.SAM2_HIERA.CHANNEL_LIST

        self.hiera = Hiera(
            embed_dim=embed_dim,           
            num_heads=num_heads,            
            stages=stages,    
            global_att_blocks=global_att_blocks, 
            window_pos_embed_bkg_spatial_size=(7, 7), 
            window_spec=window_spec,  
            drop_path_rate=drop_path_rate
        )

        guide_dim = sam2_channels[3]  
        LOFTUP_PRETRAIN_DIM = 768     
        
        self.sam_to_loftup = nn.Identity()
        if guide_dim != LOFTUP_PRETRAIN_DIM:
            self.sam_to_loftup = nn.Conv2d(guide_dim, LOFTUP_PRETRAIN_DIM, kernel_size=1)
            nn.init.kaiming_normal_(self.sam_to_loftup.weight, mode='fan_out', nonlinearity='relu')
            
        loftup_ckpt_path = "/data/xyjiang/zlni/zlni/data/loftup/loftup_dinov2b.ckpt"
        self.loftup_res3 = load_loftup_checkpoint(loftup_ckpt_path, n_dim=LOFTUP_PRETRAIN_DIM, lr_pe_type="sine", freeze=False)
        self.loftup_res4 = load_loftup_checkpoint(loftup_ckpt_path, n_dim=LOFTUP_PRETRAIN_DIM, lr_pe_type="sine", freeze=False)

        self.inject_res3 = nn.Conv2d(LOFTUP_PRETRAIN_DIM, sam2_channels[1], kernel_size=1) # 768 -> 192
        self.inject_res4 = nn.Conv2d(LOFTUP_PRETRAIN_DIM, sam2_channels[2], kernel_size=1) # 768 -> 384
        
        for m in [ self.inject_res3, self.inject_res4]:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        self.up_to_loftup_8x  = nn.Conv2d(sam2_channels[1], LOFTUP_PRETRAIN_DIM, kernel_size=1)
        self.up_to_loftup_16x = nn.Conv2d(sam2_channels[2], LOFTUP_PRETRAIN_DIM, kernel_size=1)

        for m in [self.up_to_loftup_8x, self.up_to_loftup_16x]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        self.align_res2 = nn.Conv2d(sam2_channels[0], 256,  kernel_size=1)
        self.align_res3 = nn.Conv2d(sam2_channels[1], 512,  kernel_size=1)
        self.align_res4 = nn.Conv2d(sam2_channels[2], 1024, kernel_size=1)
        self.align_res5 = nn.Conv2d(sam2_channels[3], 2048, kernel_size=1)
        
        for m in [self.align_res2, self.align_res3, self.align_res4, self.align_res5]:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None: nn.init.zeros_(m.bias)

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    def forward(self, x):
        B, C, H, W = x.shape
        
        hiera_outputs = self.hiera(x)
        sam_4x  = hiera_outputs[0]  
        sam_8x  = hiera_outputs[1]  
        sam_16x = hiera_outputs[2]  
        sam_32x = hiera_outputs[3]  
        
        #  LoftUp需要 [0, 1] RGB 原图
        #x = (img_rgb/255.0 - mean_sam2) / std_sam2 
        mean_sam2 = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std_sam2  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_rgb_01  = (x * std_sam2) + mean_sam2  # 完美回到 [0, 1] 区间
        
        # 缩小原图
        rgb_8x  = F.interpolate(x_rgb_01, size=(H//8,  W//8),  mode='area')
        rgb_16x = F.interpolate(x_rgb_01, size=(H//16, W//16), mode='area')
         # 让各层的 SAM2 升维到 768来与原图进行loftup
        hf_8x  = self.loftup_res3(lr_feats=self.up_to_loftup_8x(sam_8x),  img=rgb_8x)
        hf_16x = self.loftup_res4(lr_feats=self.up_to_loftup_16x(sam_16x), img=rgb_16x)

        fused_8x  = sam_8x  + self.inject_res3(hf_8x)  # 192 + 192
        fused_16x = sam_16x + self.inject_res4(hf_16x) # 384 + 384
        del hf_8x, hf_16x, rgb_8x, rgb_16x
        features = {}
        features["res2"] = self.align_res2(sam_4x)     # 256
        features["res3"] = self.align_res3(fused_8x)   # 512
        features["res4"] = self.align_res4(fused_16x)  # 1024
        features["res5"] = self.align_res5(sam_32x)    # 2048
        
        return features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

