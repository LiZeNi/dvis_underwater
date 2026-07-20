import os
import sys
import json
import copy
import shutil
import subprocess
import re
import numpy as np
from pycocotools import mask as mask_utils

# ==================== ⚙️ 配置区域 ====================
GT_JSON_PATH = "/data/xyjiang/zlni/zlni/data/fishtrack_ytvis/annotations/val_600.json"
PRED_JSON_PATH = "/data/xyjiang/zlni/zlni/data/UMOT_weight/dvis++/CTVIS_R50_OVIS/old_inference/results.json"

TRACKEVAL_DIR = "/data/xyjiang/zlni/zlni/TrackEval"
TEMP_DIR = "/data/xyjiang/zlni/zlni/data/temp_trackeval"

CLASS_NAMES = ["Fish", "Apparatus", "Bait", "Plants", "Crustaceans", "Other_Animals"]
# ====================================================

sys.path.append(TRACKEVAL_DIR)

# 用于只打印一次调试信息
_debug_printed = False

def safe_get_rle(seg, h, w):
    global _debug_printed
    if not seg:
        return None
    try:
        # 1. 已经是合法的 RLE 字典
        if isinstance(seg, dict) and 'counts' in seg:
            if isinstance(seg['counts'], bytes):
                seg['counts'] = seg['counts'].decode('utf-8')
            elif isinstance(seg['counts'], list):
                # counts 是未压缩的 list，直接返回，frPyObjects 也可以处理
                pass
            # 验证
            mask_utils.frPyObjects(seg, h, w)
            return seg

        # 2. 字符串（RLE 字符串 或 polygon JSON 字符串）
        if isinstance(seg, str):
            if not seg.strip():
                return None
            rles = mask_utils.frPyObjects(seg, h, w)
            rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            return rle

        # 3. 列表形式的多边形
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, h, w)
            rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            return rle

        # 4. 新增：处理 bytes 类型的 counts（有时直接存为 bytes）
        if isinstance(seg, bytes):
            seg_str = seg.decode('utf-8')
            return safe_get_rle(seg_str, h, w)  # 递归尝试字符串

        # 5. 未知类型，打印一次调试信息
        if not _debug_printed:
            preview = str(seg)[:200]
            print(f"\n[DEBUG] Unknown seg type: {type(seg)}, preview: {preview}")
            _debug_printed = True

    except Exception as e:
        # 任何解析失败都返回 None（可以根据需要打开下面的注释查看具体错误）
        # if not _debug_printed:
        #     print(f"[DEBUG] Parse error: {e}")
        #     _debug_printed = True
        pass
    return None


def prepare_trackeval_folders(gt_data, pred_data):
    global _debug_printed
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    gt_dir = os.path.join(TEMP_DIR, "gt", "youtube_vis_val")
    trk_dir = os.path.join(TEMP_DIR, "trackers", "youtube_vis_val", "MyModel")
    trk_data_dir = os.path.join(trk_dir, "data")

    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(trk_data_dir, exist_ok=True)

    # ---------- GT ----------
    gt_for_eval = copy.deepcopy(gt_data)
    vid_to_hw = {v['id']: (v['height'], v['width']) for v in gt_for_eval['videos']}
    if 'info' not in gt_for_eval:
        gt_for_eval['info'] = {"description": "FishTrack", "version": "1.0", "year": 2024}

    total_gt_masks = 0
    for ann in gt_for_eval['annotations']:
        vid_id = ann['video_id']
        h, w = vid_to_hw[vid_id]
        ann['height'] = h
        ann['width'] = w
        areas = []
        for i, seg in enumerate(ann['segmentations']):
            rle = safe_get_rle(seg, h, w)
            ann['segmentations'][i] = rle
            if rle:
                areas.append(float(mask_utils.area(rle)))
                total_gt_masks += 1
            else:
                areas.append(None)
        ann['areas'] = areas

    for i, cat in enumerate(gt_for_eval['categories']):
        if i < len(CLASS_NAMES):
            cat['name'] = CLASS_NAMES[i]

    with open(os.path.join(gt_dir, "data.json"), 'w') as f:
        json.dump(gt_for_eval, f)

    print(f"✅ GT 有效掩码总数: {total_gt_masks}")

    # ---------- 预测 ----------
    pred_for_eval = []
    total_pred_masks = 0
    missing_score_count = 0
    pred_track_ids = {id(res): idx + 1 for idx, res in enumerate(pred_data)}

    for res in pred_data:
        new_res = copy.deepcopy(res)
        new_res['id'] = pred_track_ids[id(res)]

        if 'score' not in new_res:
            new_res['score'] = 1.0
            missing_score_count += 1

        vid_id = new_res['video_id']
        if vid_id in vid_to_hw:
            h, w = vid_to_hw[vid_id]
            new_res['height'] = h
            new_res['width'] = w
            areas = []
            for i, seg in enumerate(new_res['segmentations']):
                rle = safe_get_rle(seg, h, w)
                new_res['segmentations'][i] = rle
                if rle:
                    areas.append(float(mask_utils.area(rle)))
                    total_pred_masks += 1
                else:
                    areas.append(None)
            new_res['areas'] = areas

        pred_for_eval.append(new_res)

    if missing_score_count > 0:
        print(f"⚠️ 有 {missing_score_count} 个预测缺少 'score' 字段，已自动设为 1.0")

    print(f"✅ 预测有效掩码总数: {total_pred_masks}")

    with open(os.path.join(trk_data_dir, "data.json"), 'w') as f:
        json.dump(pred_for_eval, f)


def run_trackeval():
    cmd = [
        "python", os.path.join(TRACKEVAL_DIR, "scripts", "run_youtube_vis.py"),
        "--GT_FOLDER", os.path.join(TEMP_DIR, "gt") + "/",
        "--TRACKERS_FOLDER", os.path.join(TEMP_DIR, "trackers") + "/",
        "--TRACKERS_TO_EVAL", "MyModel",
        "--SPLIT_TO_EVAL", "val",
        "--PRINT_RESULTS", "True",
        "--PRINT_ONLY_COMBINED", "True",
        "--OUTPUT_SUMMARY", "True",
        "--METRICS", "HOTA", "CLEAR", "Identity"
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 调试输出
    print("\n--- TrackEval stdout ---")
    print(result.stdout)
    if result.stderr:
        print("\n--- TrackEval stderr ---")
        print(result.stderr)

    if result.returncode != 0:
        print("TrackEval 运行失败！")
        return 0.0, 0.0, 0.0

    hota, mota, idf1 = 0.0, 0.0, 0.0
    lines = result.stdout.split('\n')

    for i, line in enumerate(lines):
        if "HOTA metrics:" in line:
            for j in range(i, len(lines)):
                if lines[j].startswith("COMBINED"):
                    parts = re.split(r'\s+', lines[j].strip())
                    hota = float(parts[1]) if len(parts) > 1 else 0.0
                    break
        elif "CLEAR metrics:" in line:
            for j in range(i, len(lines)):
                if lines[j].startswith("COMBINED"):
                    parts = re.split(r'\s+', lines[j].strip())
                    mota = float(parts[1]) if len(parts) > 1 else 0.0
                    break
        elif "Identity metrics:" in line:
            for j in range(i, len(lines)):
                if lines[j].startswith("COMBINED"):
                    parts = re.split(r'\s+', lines[j].strip())
                    idf1 = float(parts[1]) if len(parts) > 1 else 0.0
                    break

    return hota, mota, idf1


def main():
    print("🚀 加载数据...")
    with open(GT_JSON_PATH, 'r') as f:
        gt_data = json.load(f)
    with open(PRED_JSON_PATH, 'r') as f:
        pred_data = json.load(f)

    print("📁 准备 TrackEval 所需文件...")
    prepare_trackeval_folders(gt_data, pred_data)

    print("⚙️ 运行 TrackEval (类别相关评估)...")
    hota, mota, idf1 = run_trackeval()

    print("\n" + "=" * 50)
    print("🏆 类别相关 (Class-Aware) 追踪指标")
    print("=" * 50)
    print(f"  HOTA : {hota:.2f}%")
    print(f"  MOTA : {mota:.2f}%")
    print(f"  IDF1 : {idf1:.2f}%")
    print("=" * 50)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    main()