import torch
import time
import sys

def hold_gpus(target_gpus=[0, 1], memory_gb=20):
    """
    霸占指定的多个 GPU，并占用指定大小的显存。
    target_gpus: 列表，例如 [0, 1] 代表霸占 cuda:0 和 cuda:1
    memory_gb: 每张卡想占用的显存大小 (GB)
    """
    # 计算需要的张量第一维 N 
    # 算法: N * 1024 * 1024 * 4(bytes) = memory_gb * 1024 * 1024 * 1024
    # 所以 N = memory_gb * 256
    N = int(memory_gb * 256)
    
    placeholders = []
    
    print(f"🚀 开始霸占 GPU: {target_gpus}，每张卡申请 ~{memory_gb}GB 显存...")
    
    try:
        for gpu_id in target_gpus:
            device = torch.device(f'cuda:{gpu_id}')
            # 申请 20GB 张量
            # 使用 torch.empty 比 torch.zeros 更快，因为它不初始化内存里的数值
            tensor = torch.empty(N, 1024, 1024, device=device)
            placeholders.append(tensor)
            
            # 强迫显卡真正分配显存
            torch.cuda.synchronize(device)
            print(f"✅ GPU {gpu_id} 已成功驻扎 {memory_gb}GB!")
            
    except torch.cuda.OutOfMemoryError:
        print(f"\n❌ OOM 失败: 某张显卡剩余空间不足 {memory_gb}GB，无法完成霸占！")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

    print("\n🛡️ 占坑完成！现在这些显卡看起来很忙（但其实完全不吃算力）。")
    print("⏳ 按 Ctrl+C 随时释放它们并退出程序...")
    
    # 死循环挂机，维持显存占用
    try:
        while True:
            time.sleep(3600) # 睡一个小时醒来一下，几乎 0 消耗
    except KeyboardInterrupt:
        print("\n\n👋 收到退出信号，正在释放显卡...")
        # 清空列表，释放显存
        del placeholders
        torch.cuda.empty_cache()
        print("✅ 显卡已全部释放，再见！")

if __name__ == "__main__":
    # 在这里配置你想霸占的显卡 ID 和显存大小 (GB)
    hold_gpus(target_gpus=[2,3], memory_gb=30)