# MOTA_HOTA.py
# Copyright (c) 2026. All rights reserved.
# Written for DVIS_FBFL Video MOTA & HOTA Evaluation.

import json
import os
import shutil
import sys
from typing import Tuple, List, Dict, Any
import numpy as np
from pycocotools import mask as mask_utils

# 🌟 将 TrackEval 库加入到 Python 环境路径中
TRACKEVAL_PATH = "/data/zlni/zlni/TrackEval"
if TRACKEVAL_PATH not in sys.path:
    sys.path.insert(0, TRACKEVAL_PATH)

import trackeval


# =========================================================================
# ⚙️ 核心参数控制区
# =========================================================================
# 🌟 阈值控制：只评测置信度大于该值的结果。
# 设置为 0.5 可以瞬间剪断背景噪声（CLR_FP），你可以随时调整它来寻找最优指标。
CONF_THRESHOLD = 0.5  


def prepare_mot_challenge_folders(
    gt_file: str, 
    pred_file: str, 
    temp_dir: str
) -> Tuple[str, str, List[str]]:
    """
    解析 VIS 数据集并将两者转换为标准的、扁平化无嵌套的 MOTChallenge 文件目录。
    """
    gt_root = os.path.join(temp_dir, "gt")
    tracker_root = os.path.join(temp_dir, "trackers", "CTVIS_Tracker", "data")
    
    os.makedirs(gt_root, exist_ok=True)
    os.makedirs(tracker_root, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. 读取 Ground Truth 数据集 (val.json)
    # -----------------------------------------------------------------
    print(">>> 正在加载 Ground Truth JSON...")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
        
    video_id_to_meta = {v['id']: v for v in gt_data['videos']}
    seq_names = []

    print(">>> 正在转换 Ground Truth 至 MOT 格式...")
    video_anns = {vid: [] for vid in video_id_to_meta.keys()}
    for ann in gt_data['annotations']:
        video_anns[ann['video_id']].append(ann)
        
    for vid, anns in video_anns.items():
        v_meta = video_id_to_meta[vid]
        v_name = v_meta['name']
        v_len = v_meta['length']
        seq_names.append(v_name)
        
        v_gt_dir = os.path.join(gt_root, v_name, "gt")
        os.makedirs(v_gt_dir, exist_ok=True)
        
        gt_lines = []
        for track_idx, ann in enumerate(anns):
            track_id = track_idx + 1
            bboxes = ann.get('bboxes', [])
            
            for t in range(min(v_len, len(bboxes))):
                box = bboxes[t]
                if box is not None:
                    gt_lines.append(f"{t+1},{track_id},{box[0]},{box[1]},{box[2]},{box[3]},1,1,1")
                    
        with open(os.path.join(v_gt_dir, "gt.txt"), 'w') as f_gt:
            # 安全写入：如果为空，写入严格的 0 字节
            if len(gt_lines) > 0:
                f_gt.write("\n".join(gt_lines) + "\n")
            else:
                f_gt.write("")
            
        with open(os.path.join(gt_root, v_name, "seqinfo.ini"), 'w') as f_info:
            f_info.write(f"[Sequence]\n")
            f_info.write(f"name={v_name}\n")
            f_info.write(f"imDir=img1\n")
            f_info.write(f"frameRate=1\n")
            f_info.write(f"seqLength={v_len}\n")
            f_info.write(f"imWidth={v_meta['width']}\n")
            f_info.write(f"imHeight={v_meta['height']}\n")
            f_info.write(f"imExt=.jpg\n")

    # 写入 seqmap 文件
    seqmaps_dir = os.path.join(temp_dir, "gt", "seqmaps")
    os.makedirs(seqmaps_dir, exist_ok=True)
    seqmap_file_path = os.path.join(seqmaps_dir, "YTVIS_val-all.txt")
    with open(seqmap_file_path, 'w') as f_map:
        f_map.write("name\n" + "\n".join(seq_names) + "\n")

    # -----------------------------------------------------------------
    # 2. 读取并转换 CTVIS 预测结果 (results.json)
    # -----------------------------------------------------------------
    print(f">>> 正在加载 CTVIS 预测结果 JSON...")
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    print(f">>> 正在解码 RLE 掩膜并转换 Tracker 结果 (过滤置信度 < {CONF_THRESHOLD} 的背景虚警)...")
    video_preds = {vid: [] for vid in video_id_to_meta.keys()}
    for pred in pred_data:
        video_preds[pred['video_id']].append(pred)

    for vid, preds in video_preds.items():
        v_meta = video_id_to_meta[vid]
        v_name = v_meta['name']
        v_len = v_meta['length']
        
        tracker_lines = []
        for track_idx, pred in enumerate(preds):
            track_id = track_idx + 1
            score = pred.get('score', 1.0)
            
            # 核心过滤：若低于置信度，视为背景杂波丢弃
            if score < CONF_THRESHOLD:
                continue
                
            segs = pred.get('segmentations', [])
            for t in range(min(v_len, len(segs))):
                rle = segs[t]
                if rle is not None:
                    bbox = mask_utils.toBbox(rle)
                    if bbox[2] * bbox[3] >= 1.0:
                        tracker_lines.append(f"{t+1},{track_id},{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},{score:.4f},-1,-1,-1")
                        
        with open(os.path.join(tracker_root, f"{v_name}.txt"), 'w') as f_track:
            # 核心安全修复：如果该视频无预测结果，写入绝对的 0 字节，防止 Sniffer 崩溃
            if len(tracker_lines) > 0:
                f_track.write("\n".join(tracker_lines) + "\n")
            else:
                f_track.write("")

    print(">>> ✅ MOT 格式转换已顺利完成！")
    return gt_root, tracker_root, seqmap_file_path


def evaluate_via_trackeval(temp_dir: str, seqmap_file: str) -> Dict[str, Any]:
    """
    通过 Python API 加载 TrackEval 并进行计算。
    完全静音底层表格输出，将合并结果作为字典返回。
    """
    print("\n" + "="*60)
    print("  🏃 正在后台计算 TrackEval 评测指标 ...")
    print("="*60)
    
    # 1. 基础配置
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR'], 'THRESHOLD': 0.5}
    
    # -----------------------------------------------------------------
    # 🌟 调整输出级别以完全静音冗余输出
    # -----------------------------------------------------------------
    default_eval_config['PRINT_CONFIG'] = False          # ❌ 隐藏配置表格
    default_eval_config['PRINT_RESULTS'] = False         # 🚀 彻底隐藏 TrackEval 内部庞大的文本表格
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True   # ❌ 减少不必要的进度条
    default_eval_config['PLOT_CURVES'] = False            # ❌ 不进行绘图，节省计算开销
    
    default_eval_config['USE_PARALLEL'] = True  
    default_eval_config['NUM_PARALLEL_CORES'] = 8
    default_eval_config['OUTPUT_SUMMARY'] = False
    default_eval_config['OUTPUT_DETAILED'] = False
    
    default_dataset_config['GT_FOLDER'] = os.path.join(temp_dir, "gt")
    default_dataset_config['TRACKERS_FOLDER'] = os.path.join(temp_dir, "trackers")
    default_dataset_config['BENCHMARK'] = "YTVIS_val"
    default_dataset_config['SPLIT_TO_EVAL'] = "all"
    default_dataset_config['TRACKERS_TO_EVAL'] = ["CTVIS_Tracker"]
    default_dataset_config['SKIP_SPLIT_FOL'] = True
    default_dataset_config['SEQMAP_FILE'] = seqmap_file
    default_dataset_config['DO_PREPROC'] = False 
    
    # 2. 初始化 Evaluator
    evaluator = trackeval.Evaluator(default_eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(default_dataset_config)]
    
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR]:
        metrics_list.append(metric(default_metrics_config))
        
    # 3. 运行评测并获取返回结果
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    return output_res


def print_evaluation_summary(output_res: Dict[str, Any]):
    """
    从内存数据中精准提取并格式化输出关键指标。
    已根据 TrackEval 源码的真实返回键值进行嵌套对齐。
    """
    try:
        dataset_key = "MotChallenge2DBox"
        tracker_key = "CTVIS_Tracker"
        class_key = "pedestrian" # 🌟 指标归一化，单类默认使用 pedestrian
        
        # 1. 🌟 修复处：依照 TrackEval 真实字典结构，从 ['COMBINED_SEQ'][class_key] 节点提取指标
        hota_comb = output_res[dataset_key][tracker_key]['COMBINED_SEQ'][class_key]['HOTA']
        hota_val = np.mean(hota_comb['HOTA']) * 100
        deta_val = np.mean(hota_comb['DetA']) * 100
        assa_val = np.mean(hota_comb['AssA']) * 100
        detre_val = np.mean(hota_comb['DetRe']) * 100
        detpr_val = np.mean(hota_comb['DetPr']) * 100
        loca_val = np.mean(hota_comb['LocA']) * 100
        
        # 2. 🌟 修复处：依照同样结构提取 CLEAR 传统评价分量
        clear_comb = output_res[dataset_key][tracker_key]['COMBINED_SEQ'][class_key]['CLEAR']
        mota_val = clear_comb['MOTA'] * 100
        motp_val = clear_comb['MOTP'] * 100
        idsw_val = int(clear_comb['IDSW'])
        tp_val = int(clear_comb['CLR_TP'])
        fp_val = int(clear_comb['CLR_FP'])
        fn_val = int(clear_comb['CLR_FN'])
        
        print("\n" + "="*60)
        print("  📊 CTVIS 视频实例跟踪最终评测指标汇总 (COMBINED Summary)")
        print(f"  (过滤阈值 CONF_THRESHOLD = {CONF_THRESHOLD})")
        print("="*60)
        print("  [HOTA 体系指标 - 时空跟踪综合评价]")
        print(f"  • HOTA (时空跟踪综合得分)  : {hota_val:.3f}%")
        print(f"  • DetA (检测定位准确度)    : {deta_val:.3f}%")
        print(f"  • AssA (时序追踪匹配度)    : {assa_val:.3f}%")
        print(f"  • DetRe (检测召回率/Recall) : {detre_val:.3f}%")
        print(f"  • DetPr (检测精准度/Prec)   : {detpr_val:.3f}%")
        print(f"  • LocA (坐标定位精确度)    : {loca_val:.3f}%")
        print("  " + "-"*56)
        print("  [CLEAR MOT 体系指标 - 传统跟踪评价]")
        print(f"  • MOTA (多目标跟踪精度)    : {mota_val:.3f}%")
        print(f"  • MOTP (多目标跟踪定位度)  : {motp_val:.3f}%")
        print(f"  • IDSW (标识漂移/切换数)   : {idsw_val} 次")
        print(f"  • True Positives (TP)     : {tp_val}")
        print(f"  • False Positives (FP)    : {fp_val}")
        print(f"  • False Negatives (FN)    : {fn_val}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n⚠️ 提取指标汇总时发生错误: {e}")
        print("请检查结果字典结构。")


if __name__ == "__main__":
    gt_file = "/data/zlni/zlni/data/fishtrack_ytvis/annotations/val.json"
    pred_file = "/data/zlni/zlni/data/UMOT_weight/research/depth/cen/dvis++_DIONv2l/inference/results.json"
    
    # 临时评测根目录
    temp_dir = "/data/xyjiang/zlni/zlni/data/temp_eval_mot"
    
    try:
        # 执行格式化转换，获取导出的所有视频序列名称
        _, _, seqmap_file_path = prepare_mot_challenge_folders(gt_file, pred_file, temp_dir)
        
        # 执行 TrackEval 内存评测并获取评估结果
        output_res = evaluate_via_trackeval(temp_dir, seqmap_file_path)
        
        # 🌟 在终端最后输出提炼格式化后的核心指标面板
        print_evaluation_summary(output_res)
        
    finally:
        # 清理临时转换目录
        if os.path.exists(temp_dir):
            print("🧹 评测结束，正在清理临时转换目录...")
            shutil.rmtree(temp_dir)
            print("✨ 临时空间已被全部释放！")