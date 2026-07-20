import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.modeling import build_backbone
from detectron2.layers import ShapeSpec
import sys
# 确保你的项目根目录在系统路径中
sys.path.append("/data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus")

# 假设你的 SAM2HieraBackbone 写在 research/modeling/backbone/sam2_hiera.py 里
# 你需要导入它，才能激活它里面的 @BACKBONE_REGISTRY.register()
from research.modeling.backbone.sam2_hiera import SAM2HieraBackbone
# ----------------- 1. 模拟 Detectron2 配置注册 -----------------
# 假设你已经把 SAM2HieraBackbone 放在了这里，下面这行保证它被注册进 BACKBONE_REGISTRY
# from your_project.modeling.backbone.sam2_hiera import SAM2HieraBackbone 

def add_sam2_hiera_config(cfg):
    """为 Detectron2 注册 SAM 2 Hiera 的专属配置节点"""
    cfg.MODEL.SAM2_HIERA = CN()
    # 下面填写 Tiny 版本的默认值
    cfg.MODEL.SAM2_HIERA.EMBED_DIM = 112
    cfg.MODEL.SAM2_HIERA.NUM_HEADS = 2
    cfg.MODEL.SAM2_HIERA.STAGES = [1, 2, 7, 2]
    cfg.MODEL.SAM2_HIERA.GLOBAL_ATT_BLOCKS = [5, 7, 9]
    cfg.MODEL.SAM2_HIERA.WINDOW_SPEC = [8, 4, 14, 7]
    cfg.MODEL.SAM2_HIERA.DROP_PATH_RATE = 0.1

def setup_test_cfg():
    cfg = get_cfg()
    add_sam2_hiera_config(cfg)
    
    # 强制指定使用你刚写的 Backbone
    cfg.MODEL.BACKBONE.NAME = "SAM2HieraBackbone"
    return cfg

# ----------------- 2. 主测试函数 -----------------
def test_backbone():
    print("="*60)
    print("🚀 启动 Detectron2 Registry 架构探针 (SAM2 Hiera)...")
    print("="*60)
    
    cfg = setup_test_cfg()
    
    # 🚨 这里是见证奇迹的时刻：看系统能不能根据 Config 自动把你的模型造出来
    try:
        model = build_backbone(cfg)
        print("✅ 成功通过 build_backbone 实例化模型！")
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        return

    # 模拟输入 (Batch=2, Channel=3, Height=512, Width=1024)
    B, C, H, W = 2, 3, 512, 1024
    dummy_input = torch.randn(B, C, H, W, requires_grad=True)
    print(f"\n📥 输入张量 (Images) 形状: {dummy_input.shape}  [B, C, H, W]")
    
    # 获取模型自己声明的形状承诺
    expected_shapes = model.output_shape()
    
    # 跑一次前向传播拿真实的 Tensor
    features = model(dummy_input)
    
    print("\n📤 输出特征金字塔 (Features) 解析对比:")
    all_correct = True
    
    for level_name in ["res2", "res3", "res4", "res5"]:
        if level_name not in features:
            print(f"❌ 缺失 {level_name} 层！")
            all_correct = False
            continue
            
        feat = features[level_name]
        spec = expected_shapes[level_name]
        
        # 反推实际下采样率
        actual_stride_h = H // feat.shape[2]
        
        print(f"\n   [{level_name.upper()}] 提取到的 Tensor 形状: {list(feat.shape)}")
        print(f"      ↳ [承诺] 通道数: {spec.channels} | [真实] 通道数: {feat.shape[1]}")
        print(f"      ↳ [承诺] Stride: {spec.stride}x | [真实] Stride: {actual_stride_h}x")
        
        if feat.shape[1] != spec.channels:
            print(f"      ❌ 警告: 通道数不匹配！(会导致 MaskFormerHead 的 input_proj 报错)")
            all_correct = False
            
        if actual_stride_h != spec.stride:
            print(f"      ❌ 警告: 空间分辨率下采样率不符合预期！(会导致 Deformable Attention 坐标崩塌)")
            all_correct = False

        if not feat.requires_grad:
            print(f"      ❌ 警告: 该层没有梯度，模型无法微调！")
            all_correct = False

    print("\n" + "="*60)
    if all_correct:
        print("🎉 完美！Hiera 输出的每一层通道数和物理 Stride 比例完全符合预期！")
        print("这套通过 Config 实例化的组件已经准备好投入端到端训练了！")
    else:
        print("⚠️ 依然存在维度或尺度未对齐的现象，请检查 Hiera 原生的 patch_embed 或 q_pool 逻辑！")
    print("="*60)

if __name__ == "__main__":
    test_backbone()