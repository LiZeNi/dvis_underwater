#!/bin/bash

# ================= 配置区 =================
# 你要苦苦等待的那个权重文件路径
TARGET_FILE="/data/xyjiang/zlni/zlni/data/UMOT_weight/dvis++/CTVIS_R50_OVIS/model_final.pth"

# 阶段二要执行的脚本路径
NEXT_SCRIPT="/data/xyjiang/zlni/zlni/DVIS_Plus/train/train_dvis_daq.sh"

# 每隔多少秒去偷看一次？(这里设置为 300 秒，即 5 分钟)
CHECK_INTERVAL=300
# ==========================================

echo "=========================================================="
echo "👀 开始监控模式... 正在等待第一阶段模型生成..."
echo "📂 目标文件: $TARGET_FILE"
echo "⏱️ 每 $CHECK_INTERVAL 秒检查一次..."
echo "=========================================================="

# 核心逻辑：无限循环，直到文件出现才打破循环
while [ ! -f "$TARGET_FILE" ]; do
    # 打印当前时间，证明脚本还活着
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 还没找到文件，继续等待..."
    
    # 挂起睡眠，不浪费任何 CPU 资源
    sleep $CHECK_INTERVAL
done

echo "=========================================================="
echo "🎉 [$(date +'%Y-%m-%d %H:%M:%S')] 万岁！找到权重文件啦！"
echo "🚀 立刻启动阶段二训练: $NEXT_SCRIPT"
echo "=========================================================="

# 切换到你跑 DAQ 需要的那个工作目录（极其关键）
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_DAQ

# 正式拉起你的第二阶段脚本
bash $NEXT_SCRIPT