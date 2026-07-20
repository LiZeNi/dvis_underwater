import torch
import os

# ==================== 配置路径 ====================
# 1. 你的 SAM 2 预训练权重路径 (提供骨干知识)
SAM2_CKPT_PATH = "/data/xyjiang/zlni/zlni/data/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

# 2. 你的 DVIS / Mask2Former 预训练权重路径 (提供 Decoder 和 Tracker 知识)
DVIS_CKPT_PATH = "/data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_r50_ovis.pth"

# 3. 缝合后的新权重保存路径 (用于配置在 YAML 的 MODEL.WEIGHTS)
OUTPUT_CKPT_PATH = "/data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/CTVIS_SAM2_base_OVIS_untrain.pth"
# ==================================================

def main():
    print("🚀 开始进行跨架构权重缝合手术...")

    # 1. 加载双方的权重
    print(f"📥 正在读取 SAM 2 权重: {SAM2_CKPT_PATH}")
    sam2_dict = torch.load(SAM2_CKPT_PATH, map_location="cpu")["model"]
    
    print(f"📥 正在读取 DVIS 权重: {DVIS_CKPT_PATH}")
    dvis_checkpoint = torch.load(DVIS_CKPT_PATH, map_location="cpu")
    dvis_model_dict = dvis_checkpoint["model"]

    # 2. 剥离 DVIS 里老旧的 ResNet/Swin 骨干网络参数
    # 如果原模型用的是 ResNet，参数通常以 "backbone." 开头
    # 这一步不是必须的，但为了保持权重文件干净小巧，我们把它们删掉
    keys_to_delete = [k for k in dvis_model_dict.keys() if k.startswith("backbone.")]
    for k in keys_to_delete:
        del dvis_model_dict[k]
    print(f"✂️ 已剔除 DVIS 原有骨干参数，共 {len(keys_to_delete)} 个 tensor。")

    # 3. 提取并映射 SAM 2 的参数到 DVIS 的 Backbone
    # SAM 2 的核心骨干保存在 "image_encoder.trunk." 下
    # 我们的 Detectron2 模型里，骨干对象叫 self.backbone，里面实例化了 self.hiera
    # 所以我们要把前缀替换为 "backbone.hiera."
    mapped_sam2_params = 0
    for k, v in sam2_dict.items():
        if k.startswith("image_encoder.trunk."):
            # 去掉 SAM 2 的外壳前缀，换上 Detectron2 的前缀
            new_key = k.replace("image_encoder.trunk.", "backbone.hiera.")
            
            # 将映射好的权重塞入 DVIS 的字典中
            dvis_model_dict[new_key] = v
            mapped_sam2_params += 1

    print(f"💉 已成功注入 SAM 2 参数，共 {mapped_sam2_params} 个 tensor。")

    # 4. 消除优化器状态和迭代步数
    # 这是关键！因为我们更换了骨干，不能继续用旧的优化器状态，必须重新训练 (Finetune)
    if "optimizer" in dvis_checkpoint:
        del dvis_checkpoint["optimizer"]
    if "iteration" in dvis_checkpoint:
        dvis_checkpoint["iteration"] = 0
        
    print("🧹 已清空旧的优化器状态和训练步数，准备全新微调。")

    # 5. 保存新的“弗兰肯斯坦”究极权重
    os.makedirs(os.path.dirname(OUTPUT_CKPT_PATH), exist_ok=True)
    torch.save(dvis_checkpoint, OUTPUT_CKPT_PATH)
    
    print("="*60)
    print(f"🎉 缝合大成功！")
    print(f"请在你的 YAML 配置文件中修改如下：\nMODEL:\n  WEIGHTS: '{OUTPUT_CKPT_PATH}'")
    print("="*60)

if __name__ == "__main__":
    main()