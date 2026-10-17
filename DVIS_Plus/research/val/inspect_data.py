# /data/xyjiang/zlni/zlni/inspect_vis_data.py
import json
import os

def inspect_vis_file(filepath: str, label: str):
    print("\n" + "="*60)
    print(f"  🔍 正在分析 [{label}] 结构 ...")
    print(f"  路径: {filepath}")
    print("="*60)
    
    if not os.path.exists(filepath):
        print(f"❌ 错误: 未能找到该文件，请检查路径是否正确。")
        return
        
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"📂 文件大小: {file_size_mb:.2f} MB")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    print(f"📊 数据类型: {type(data)}")
    
    # -----------------------------------------------------------------
    # 情况一：如果是字典结构（通常是 Ground Truth val.json）
    # -----------------------------------------------------------------
    if isinstance(data, dict):
        print(f"🔑 顶层键名 (Keys): {list(data.keys())}")
        
        # 1. 分析分类信息
        if "categories" in data:
            print(f"🏷️ 分类数量 (Categories): {len(data['categories'])}")
            print(f"   详细映射: {data['categories']}")
            
        # 2. 分析视频元数据
        if "videos" in data:
            print(f"🎥 视频数量 (Videos): {len(data['videos'])}")
            if len(data['videos']) > 0:
                print(f"   视频条目示例 (前向 1 个):")
                print(f"   {data['videos'][0]}")
                
        # 3. 分析真值标注信息
        if "annotations" in data:
            print(f"✍️ 标注数量 (Annotations): {len(data['annotations'])}")
            if len(data['annotations']) > 0:
                sample_ann = data['annotations'][0]
                # 过滤庞大的 RLE/Mask 数组，仅显示结构信息
                clean_ann = {k: v for k, v in sample_ann.items() if k not in ['segmentations', 'bboxes', 'areas']}
                if 'segmentations' in sample_ann:
                    clean_ann['segmentations_type'] = str(type(sample_ann['segmentations']))
                    clean_ann['segmentations_length'] = len(sample_ann['segmentations'])
                if 'bboxes' in sample_ann:
                    clean_ann['bboxes_type'] = str(type(sample_ann['bboxes']))
                    clean_ann['bboxes_length'] = len(sample_ann['bboxes'])
                    # 打印非空的样例 Bbox 以确认格式
                    non_empty_box = [b for b in sample_ann['bboxes'] if b is not None]
                    clean_ann['sample_bbox_value'] = non_empty_box[0] if len(non_empty_box) > 0 else "All None"
                print(f"   标注条目示例 (已过滤大体积 Mask 数据):")
                print(f"   {clean_ann}")
                
    # -----------------------------------------------------------------
    # 情况二：如果是列表结构（通常是 CTVIS 模型推理结果 results.json）
    # -----------------------------------------------------------------
    elif isinstance(data, list):
        print(f"📦 预测实例总数 (Predictions): {len(data)}")
        if len(data) > 0:
            sample_pred = data[0]
            clean_pred = {k: v for k, v in sample_pred.items() if k not in ['segmentations', 'bboxes']}
            if 'segmentations' in sample_pred:
                clean_pred['segmentations_type'] = str(type(sample_pred['segmentations']))
                clean_pred['segmentations_length'] = len(sample_pred['segmentations'])
                if len(sample_pred['segmentations']) > 0:
                    # 观察首帧分割结果的表示形式是 RLE 字典、Polygon 列表还是二进制
                    sample_mask = sample_pred['segmentations'][0]
                    if isinstance(sample_mask, dict):
                        clean_pred['sample_mask_structure'] = {k: type(v).__name__ for k, v in sample_mask.items()}
                    else:
                        clean_pred['sample_mask_structure'] = type(sample_mask).__name__
            print(f"   预测条目示例 (已过滤大体积 Mask 数据):")
            print(f"   {clean_pred}")
            
            # 统计视频覆盖率
            video_ids = set([p.get('video_id', None) for p in data])
            print(f"🎬 预测结果覆盖的视频 ID 数: {len(video_ids)}")
            
    else:
        print("⚠️ 未知的数据顶层结构。")

if __name__ == "__main__":
    gt_file = "/data/xyjiang/zlni/zlni/data/fishtrack_ytvis/annotations/val.json"
    pred_file = "/data/xyjiang/zlni/zlni/data/UMOT_weight/research/anyup/DINOv2_freeze_stage1/CTVIS/inference/results.json"
    
    inspect_vis_file(gt_file, "Ground Truth (val.json)")
    inspect_vis_file(pred_file, "CTVIS 推理结果 (results.json)")