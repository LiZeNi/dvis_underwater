import torch
import sys

def debug_checkpoint(ckpt_path):
    print(f"==================================================")
    print(f"🔍 Inspecting Checkpoint: {ckpt_path}")
    print(f"==================================================")
    
    try:
        # 1. 加载权重文件
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load {ckpt_path}\nError: {e}")
        return

    # 2. 检查字典的外层结构（剥壳）
    print("\n[Level 1] Top-level keys in the checkpoint:")
    if isinstance(checkpoint, dict):
        print(" -> " + ", ".join(checkpoint.keys()))
        
        # 剥壳提取真正的模型权重
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print(" -> Found 'model' key. Extracted inner state_dict.")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print(" -> Found 'state_dict' key. Extracted inner state_dict.")
        else:
            state_dict = checkpoint
            print(" -> No wrapper found. Assuming top-level is state_dict.")
    else:
        print(" -> Checkpoint is not a dictionary!")
        return

    # 3. 统计和打印最内层权重的 Key 名字
    print(f"\n[Level 2] Inner state_dict has {len(state_dict)} keys.")
    
    keys = list(state_dict.keys())
    
    print("\n[Preview] First 15 keys in the state_dict:")
    for k in keys[:15]:
        print(f"  - {k}")
        
    print("\n[Search] Checking for ViT-Adapter specific keys...")
    target_key_1 = 'patch_embed.proj.weight'
    target_key_2 = 'backbone.patch_embed.proj.weight'
    target_key_3 = 'backbone.net.patch_embed.proj.weight'
    
    if target_key_1 in state_dict:
        print(f"  ✅ FOUND raw ViT key: '{target_key_1}'")
    else:
        print(f"  ❌ NOT FOUND raw ViT key: '{target_key_1}'")
        
    if target_key_2 in state_dict:
        print(f"  ✅ FOUND wrapped ViT key: '{target_key_2}'")
    elif target_key_3 in state_dict:
        print(f"  ✅ FOUND wrapped ViT key: '{target_key_3}'")

    print(f"==================================================\n")

if __name__ == "__main__":
    # 测试导致报错的那个权重
    vit_weight_path = "/data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/online_vitl_ovis_496.pth"
    debug_checkpoint(vit_weight_path)
    
    # 也可以测试另外一个权重看看
    model_weight_path = "/data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_vitl_ovis.pth"
    debug_checkpoint(model_weight_path)