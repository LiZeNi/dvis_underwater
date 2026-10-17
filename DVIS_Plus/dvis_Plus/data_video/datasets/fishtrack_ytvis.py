# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_fishtrack_ytvis_instances"]

# FISHTRACK_CLASSES = [
#     "lutjanus_campechanus", "micropterus_salmoides", "unspecified_fish", "stenotomus_caprinus",
#     "rhomboplites_aurorubens", "pagrus_pagrus", "epinephelus_morio", "ioglossus",
#     "non_fish_bait", "malacanthus_plumieri", "etelis_coruscans", "serranus_annularis",
#     "balistes_capriscus", "mycteroperca_phenax", "calamus_nodosus", "lutjanus_griseus",
#     "anomura", "pristipomoides_aquilonaris", "calamus_leucosteus", "pristigenys_alta",
#     "chaetodon_sedentarius", "serranus_phoebe", "holacanthus_bermudensis", "mycteroperca_microlepis",
#     "haemulon_aurolineatum", "selene_vomer", "seriola_dumerili", "pterois",
#     "halichoeres_bivittatus", "seriola_rivoliana", "trachurus_trachurus", "non_fish_plant",
#     "aphareus_rutilans", "centropristis_philadelphica", "non_fish_bait_apparatus", "opistognathus",
#     "chaetodon_ocellatus", "caranx_crysos", "lachnolaimus_maximus", "serranus_subligarius",
#     "lutjanus_synagris", "calamus_proridens", "pristipomoides_zonatus", "pristipomoides_filamentosus",
#     "holanthius_martinicensis", "canthigaster_rostratus", "caulolatilus_cyanops", "seriola",
#     "pomacentridae", "pomacanthus_arcuatus", "carcharhinus_falciformis", "lutjanus_buccanela",
#     "chaetodon", "dermatolepis_inermis", "epinephelus_flavolimbatus", "pristipomoides_sieboldii",
#     "calamus_bajonado", "seriola_zonata", "bodianus_rufus", "holocentrus_adscensionis",
#     "hyporthodus_quernus", "seriola_fasciata", "halichoeres", "equetus_umbrosus",
#     "serranus", "centropristis_ocyura", "mycteroperca_interstitialis", "mullidae",
#     "pristipomoides_auricilla", "sparidae", "diplectrum_formosum", "archosargus_probatocephalus",
#     "bodianus_pulchellus", "gymnothorax_saxicola", "naso", "stenocionops_spinimana",
#     "mycteroperca_bonaci", "upeneus_parvus", "decodon_puellaris", "ocyurus_chrysurus",
#     "calamus", "paranthias_furcifer", "rypticus_maculatus", "lutjanus_vivanus",
#     "urophycis_regia", "caulolatilus_chrysops", "gymnothorax_moringa", "sphyrna_lewini",
#     "gobiidae", "serranus_atrobranchus", "sardina_pilchardus", "haemulon_melanurum",
#     "non_fish_animal", "canthidermis_sufflamen", "etelis_carbunculus", "carcharhinus",
#     "synodontidae", "epinephelus_adscensionis", "balistes_vetula", "opistognathus_aurifrons",
#     "chaetodipterus_faber", "caranx_ruber", "pomacanthus_paru", "muraena_retifera",
#     "ophichthus_puncticeps", "anthiinae", "halichoeres_bathyphilus", "cephalopholis_cruentata",
#     "mycteroperca", "liopropoma_eukrines", "engraulis_encrasicolus", "caranx_bartholomaei",
#     "syacium", "carcharhinus_plumbeus", "alectis_ciliaris", "holocentrus",
#     "lutjanus_analis", "lutjanus", "holacanthus", "epinephelus_nigritus",
#     "pomacentrus_partitus", "hypoplectrus", "decapterus", "chromis_enchrysurus",
#     "carangidae", "caranx_melampygus", "acanthurus", "xanthichthys_ringens",
#     "albula_vulpes", "sphyraena_barracuda", "rachycentron_canadum", "hypoplectrus_unicolor",
#     "acanthurus_coeruleus", "lactophrys_trigonus", "chromis_insolata", "thalassoma_bifasciatum",
#     "scomberomorus", "chromis_insolatus", "sparisoma_aurofrenatum", "scaridae",
#     "halichoeres_garnoti", "pseudupeneus_maculatus", "epinephelus", "priacanthus_arenatus",
#     "pomacanthus", "gonioplectrus_hispanus", "chromis", "equetus_lanceolatus",
#     "diodontidae", "haemulon_macrostomum", "diplectrum", "scarus_vetula",
#     "carcharhinus_perezi", "lutjanus_apodus", "chaetodon_aculeatus"
# ]

FISHTRACK_CLASSES = [
    "Fish",           # id: 1
    "Apparatus",      # id: 2 (人工装置，如铁笼、支架等)
    "Bait",           # id: 3 (诱饵肉块)
    "Plants",         # id: 4 (水生植物、海藻等)
    "Crustaceans",    # id: 5 (甲壳类，如寄居蟹、蜘蛛蟹等)
    "Other Animals",  # id: 6 (其他非鱼类动物，如海龟、水母等)
]



def _get_fishtrack_instances_meta():
    """生成 FishTrack 数据集的元数据"""
    thing_classes = FISHTRACK_CLASSES
    # 自动生成颜色 (随机颜色)
    # 如果需要固定颜色，可以手动定义一个长列表
    import random
    random.seed(42) # 固定种子保证每次颜色一致
    thing_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(thing_classes))]
    
    # 映射关系:
    # 数据集中的 category_id 是从 1 开始的 (1~146)
    # Detectron2 需要连续的 id 从 0 开始 (0~145)
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(thing_classes))}
    
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

# ============================================================================
# 2. 核心加载函数 (保持原样，微调日志)
# ============================================================================

def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, depth_root=None):
    from pycocotools.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        
        # 自动校正：如果 JSON 中的类别 ID 和 Metadata 不一致，尝试自动映射
        if hasattr(meta, "thing_dataset_id_to_contiguous_id"):
            id_map = meta.thing_dataset_id_to_contiguous_id
        else:
            # Fallback logic
            cats = ytvis_api.loadCats(cat_ids)
            thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
            meta.thing_classes = thing_classes
            id_map = {v: i for i, v in enumerate(cat_ids)}
            meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []
    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        # 这里拼接路径：image_root + video_name/frame.jpg
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        if depth_root is not None:
            # 例如：原图路径 .../images/train/video_001/000001.jpg -> 深度路径 .../depth/train/video_001/000001_depth.npy
            depth_file_names = []
            for img_path in record["file_names"]:
                # 从图像路径推断深度路径：替换 images 为 depth，并添加 _depth.npy
                rel_path = os.path.relpath(img_path, image_root)
                depth_rel_path = os.path.splitext(rel_path)[0] + "_depth.npy"
                depth_file_names.append(os.path.join(depth_root, depth_rel_path))
            record["depth_file_names"] = depth_file_names
        else:
            record["depth_file_names"] = []

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                # 只有当该帧有 bbox 和 mask 时才保留
                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                # 处理 RLE 或 Polygon
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    # 将 JSON 中的 ID (1-146) 映射到 Detectron2 的 ID (0-145)
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    return dataset_dicts

def register_fishtrack_ytvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for instance tracking.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))

    # 2. Add metadata
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )


def register_fishtrack_ytvis_depth_instances(name, metadata, json_file, image_root, depth_root):
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name, depth_root=depth_root))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, depth_root=depth_root, evaluator_type="ytvis", **metadata
    )

# ============================================================================
# 3. 主程序入口：注册并测试 FishTrack 数据集
# ============================================================================

if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets 
    from PIL import Image

    logger = setup_logger(name=__name__)

    # --- 配置路径 ---
    # 你的数据集根目录
    DATASET_ROOT = "/data/zlni/zlni/data/fishtrack_ytvis"
    DEPTH_ROOT = "/data/zlni/zlni/data/fishtrack_ytvis/depth"
    # --- 注册 Train 和 Val ---
    splits = ["train", "val"]
    for split in splits:
        name = f"fishtrack_ytvis_{split}"
        # JSON 路径: .../annotations/train.json
        json_file = os.path.join(DATASET_ROOT, "annotations", f"{split}.json")
        # 图片根路径: .../JPEGImages/train (因为 JSON 里的 file_names 是 video/xxx.jpg)
        image_root = os.path.join(DATASET_ROOT, "images", split)
        
        # 执行注册
        register_fishtrack_ytvis_instances(name, _get_fishtrack_instances_meta(), json_file, image_root)
        print(f"Successfully registered: {name}")
        
        # 深度数据集注册
        name_depth = f"fishtrack_ytvis_depth_{split}"
        depth_root = os.path.join(DEPTH_ROOT, split)   # 深度图按 split 存放
        register_fishtrack_ytvis_depth_instances(name_depth, _get_fishtrack_instances_meta(), json_file, image_root, depth_root)
        print(f"Successfully registered: {name_depth}")