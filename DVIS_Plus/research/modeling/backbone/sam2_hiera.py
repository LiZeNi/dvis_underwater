import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from ..sam2_utils.hieradet import Hiera
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec

@BACKBONE_REGISTRY.register()
class SAM2HieraBackbone(Backbone):
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
        self.align_res2 = nn.Conv2d(sam2_channels[0], 256,  kernel_size=1)
        self.align_res3 = nn.Conv2d(sam2_channels[1], 512,  kernel_size=1)
        self.align_res4 = nn.Conv2d(sam2_channels[2], 1024, kernel_size=1)
        self.align_res5 = nn.Conv2d(sam2_channels[3], 2048, kernel_size=1)
        
        for m in [self.align_res2, self.align_res3, self.align_res4, self.align_res5]:
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

    def forward(self, x):
        hiera_outputs = self.hiera(x)
        
        features = {}
        features["res2"] = self.align_res2(hiera_outputs[0])  
        features["res3"] = self.align_res3(hiera_outputs[1])  
        features["res4"] = self.align_res4(hiera_outputs[2])  
        features["res5"] = self.align_res5(hiera_outputs[3])  
        
        return features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

