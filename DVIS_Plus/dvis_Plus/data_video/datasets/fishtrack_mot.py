"""
FishTrack MOT 数据集注册
每个视频有独立的 COCO 格式 JSON 标注文件
"""

import os
import json
import logging
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

logger = logging.getLogger(__name__)

__all__ = ["register_all_fishtrack_mot"]


# ===== 1. 定义类别信息 =====
FISHTRACK_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"},
]


def _get_fishtrack_mot_instances_meta():
    """获取 FishTrack MOT 数据集元数据"""
    thing_ids = [k["id"] for k in FISHTRACK_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in FISHTRACK_CATEGORIES if k["isthing"] == 1]
    thing_classes = [k["name"] for k in FISHTRACK_CATEGORIES if k["isthing"] == 1]
    
    # ID 映射：原始 ID → 连续 ID [0, N-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


# ===== 2. 数据加载函数 =====
def load_fishtrack_mot_json(annotation_dir, image_dir, dataset_name=None):
    """
    从多个 COCO JSON 文件加载 FishTrack MOT 数据
    
    Args:
        annotation_dir: 标注目录（包含多个 JSON 文件）
        image_dir: 图像根目录（包含多个视频子目录）
        dataset_name: 数据集名称
    
    Returns:
        dataset_dicts: 视频格式的数据字典列表
    """
    from fvcore.common.timer import Timer
    
    timer = Timer()
    
    # 获取所有 JSON 文件
    json_files = sorted(Path(annotation_dir).glob("*.json"))
    
    if len(json_files) == 0:
        logger.warning(f"⚠️  No JSON files found in {annotation_dir}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files in {annotation_dir}")
    
    # 获取元数据和 ID 映射
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = [k["id"] for k in FISHTRACK_CATEGORIES]
        id_map = {k: i for i, k in enumerate(cat_ids)}
    
    # 处理每个视频
    dataset_dicts = []
    num_instances_without_valid_segmentation = 0
    
    for video_idx, json_file in enumerate(json_files):
        video_name = json_file.stem  # 文件名（不含扩展名）
        video_image_dir = os.path.join(image_dir, video_name)
        
        # 检查图像目录是否存在
        if not os.path.exists(video_image_dir):
            logger.warning(f"⚠️  Image directory not found: {video_image_dir}, skipping {video_name}")
            continue
        
        # 加载 JSON
        try:
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading {json_file}: {e}")
            continue
        
        # 检查必要字段
        if "images" not in coco_data or "annotations" not in coco_data:
            logger.warning(f"⚠️  Invalid JSON format in {json_file}, skipping")
            continue
        
        # 构建图像字典
        images_dict = {img["id"]: img for img in coco_data["images"]}
        
        if len(images_dict) == 0:
            logger.warning(f"⚠️  No images in {json_file}, skipping")
            continue
        
        # 按图像 ID 组织标注
        from collections import defaultdict
        img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        
        # 按文件名排序图像
        sorted_images = sorted(images_dict.items(), key=lambda x: x[1]["file_name"])
        
        # 创建视频记录
        record = {}
        record["file_names"] = []
        record["video_id"] = video_idx
        record["video_name"] = video_name
        
        # 收集所有帧的路径
        first_img = None
        for img_id, img_info in sorted_images:
            # 图像路径：image_dir/video_name/frame_xxx.jpg
            img_path = os.path.join(video_image_dir, img_info["file_name"])
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                logger.warning(f"⚠️  Image not found: {img_path}")
                continue
            
            record["file_names"].append(img_path)
            
            if first_img is None:
                first_img = img_info
        
        if len(record["file_names"]) == 0:
            logger.warning(f"⚠️  No valid images for video {video_name}, skipping")
            continue
        
        # 设置视频属性
        record["height"] = first_img["height"]
        record["width"] = first_img["width"]
        record["length"] = len(record["file_names"])
        
        # 处理每一帧的标注
        video_objs = []
        for img_id, img_info in sorted_images:
            img_path = os.path.join(video_image_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            
            frame_objs = []
            
            for ann in img_to_anns[img_id]:
                obj = {}
                
                # Bounding box
                bbox = ann.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                
                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                
                # Segmentation
                segm = ann.get("segmentation", None)
                if segm:
                    if isinstance(segm, dict):
                        # RLE 格式（已压缩）
                        if isinstance(segm["counts"], list):
                            # 未压缩的 RLE，需要压缩
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                        # 已经是压缩格式，直接使用
                        obj["segmentation"] = segm
                    elif isinstance(segm, list):
                        # Polygon 格式
                        valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(valid_segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue
                        obj["segmentation"] = valid_segm
                    else:
                        continue
                else:
                    # 没有分割标注，跳过
                    continue
                
                # Category
                obj["category_id"] = ann["category_id"]
                if id_map:
                    obj["category_id"] = id_map.get(obj["category_id"], 0)
                
                # 其他字段
                obj["iscrowd"] = ann.get("iscrowd", 0)
                obj["id"] = ann["id"]
                obj["area"] = ann.get("area", 0)
                
                # 如果有 track_id，保存它
                if "track_id" in ann:
                    obj["track_id"] = ann["track_id"]
                
                frame_objs.append(obj)
            
            video_objs.append(frame_objs)
        
        record["annotations"] = video_objs
        dataset_dicts.append(record)
        
        logger.info(f"  ✅ Loaded video {video_name}: {record['length']} frames, "
                   f"{sum(len(f) for f in video_objs)} annotations")
    
    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            f"⚠️  Filtered out {num_instances_without_valid_segmentation} instances "
            "without valid segmentation."
        )
    
    logger.info(f"✅ Loaded {len(dataset_dicts)} videos in {timer.seconds():.2f} seconds")
    
    return dataset_dicts


# ===== 3. 注册函数 =====
def register_fishtrack_mot_instances(name, metadata, annotation_dir, image_dir, json_file):
    """
    注册 FishTrack MOT 数据集（遵循 YTVIS 的注册模式）
    
    Args:
        name: 数据集名称
        metadata: 元数据字典
        annotation_dir: 标注目录（包含多个 JSON 文件）
        image_dir: 图像根目录（包含多个视频子目录）
        json_file: 合并的 YTVIS 格式 JSON 文件路径
    """
    # 支持重复注册
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)
    
    # 注册数据加载函数
    DatasetCatalog.register(
        name,
        lambda annotation_dir=annotation_dir, image_dir=image_dir, dataset_name=name:
            load_fishtrack_mot_json(annotation_dir, image_dir, dataset_name)
    )

    # 注册元数据（添加 json_file 字段）
    MetadataCatalog.get(name).set(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        json_file=json_file,  # ← 添加这个字段用于评估
        evaluator_type="ytvis",  # 使用 YTVIS 评估器
        **metadata
    )


# ===== 4. 预定义的数据集分割 =====
_PREDEFINED_SPLITS_FISHTRACK_MOT = {
    "fishtrack_mot_train": {
        "image_root": "image/train",
        "annotation_root": "mask_annotations/train",
        "json_file": "YTVIS_annotations/train/instances.json"  # 合并的 JSON 文件
    },
    "fishtrack_mot_val": {
        "image_root": "image/val",
        "annotation_root": "mask_annotations/val",
        "json_file": "YTVIS_annotations/val/instances.json"  # 合并的 JSON 文件
    },
}


# ===== 5. 批量注册函数 =====
def register_all_fishtrack_mot(root):
    """
    批量注册 FishTrack MOT 数据集
    
    Args:
        root: 数据集根目录
    """
    for key, paths in _PREDEFINED_SPLITS_FISHTRACK_MOT.items():
        annotation_dir = os.path.join(root, paths["annotation_root"])
        image_dir = os.path.join(root, paths["image_root"])
        json_file = os.path.join(root, paths["json_file"])
        
        # 检查目录是否存在
        if not os.path.exists(annotation_dir):
            logger.warning(f"⚠️  Annotation directory not found: {annotation_dir}")
            continue
        
        if not os.path.exists(image_dir):
            logger.warning(f"⚠️  Image directory not found: {image_dir}")
            continue
        
        # 检查合并的 JSON 文件是否存在
        if not os.path.exists(json_file):
            logger.error(f"❌ Merged JSON file not found: {json_file}")
            logger.error(f"   Please run: python create_merged_json.py --annotation_dir {annotation_dir} --output {json_file}")
            
        
        # 注册数据集
        register_fishtrack_mot_instances(
            key,
            _get_fishtrack_mot_instances_meta(),
            annotation_dir,
            image_dir,
            json_file,  # ← 传入 json_file 参数
        )
        
        logger.info(f"✅ Registered dataset: {key}")
        logger.info(f"   JSON file: {json_file}")


# ===== 6. 模块导入时自动注册 =====
if __name__.endswith(".fishtrack_mot"):
    # 直接使用绝对路径
    _root = "/home/zlni/zlni/data/fishtrack_mot"
    register_all_fishtrack_mot(_root)