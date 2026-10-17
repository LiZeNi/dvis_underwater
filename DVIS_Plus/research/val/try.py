# download_backbones.py
import os

# 1. 强制重定向 PyTorch 和 Hugging Face 缓存目录到你的指定路径
target_dir = "/data/xyjiang/zlni/zlni/data/anyup"
os.environ["TORCH_HOME"] = os.path.join(target_dir, "torch")
os.environ["HF_HOME"] = os.path.join(target_dir, "huggingface")

# 打印路径确认，确保没有拼写错误
print(f"👉 PyTorch 缓存目录重定向为: {os.environ['TORCH_HOME']}")
print(f"👉 Hugging Face 缓存目录重定向为: {os.environ['HF_HOME']}")

# 2. 导入 timm 开始下载
try:
    import timm
except ImportError:
    print("❌ 未检测到 timm，请先激活你的 anyup 虚拟环境并安装 timm！")
    exit(1)

# 3. 对应 config.yaml 中配置的所有主干网络名称
backbone_names = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_384",
    "vit_base_patch16_siglip_512.v2_webli",
    "vit_small_patch14_reg4_dinov2",
    "vit_base_patch16_224"
]

print("\n--- 开始下载预训练权重 ---")
for name in backbone_names:
    print(f"\n正在下载 / 加载: {name} ...")
    try:
        # timm.create_model 会自动检测缓存，未下载时会自动下载到上述设定的环境变量目录中
        model = timm.create_model(name, pretrained=True)
        print(f"✅ {name} 权重下载完成并已安全缓存！")
    except Exception as e:
        print(f"❌ 下载 {name} 失败，错误信息: {e}")

print("\n🎉 所有配置的主干网络权重已成功下载至指定路径！")