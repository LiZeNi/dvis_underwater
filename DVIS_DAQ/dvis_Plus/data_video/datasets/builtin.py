# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances
from .fishtrack_ytvis import register_fishtrack_ytvis_instances

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
)



# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

_PREDEFINED_SPLITS_YTVIS_2022 = {
    "ytvis_2022_val": ("ytvis_2022/valid/JPEGImages",
                       "ytvis_2022/valid/instances.json"),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                         "ovis/annotations/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                       "ovis/annotations/annotations_valid.json"),
    "ovis_test": ("ovis/test",
                        "ovis/annotations/annotations_test.json"),
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}

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



def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ytvis_2022(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2022.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_coco_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

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

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    #register_all_ytvis_2019(_root)
    #register_all_ytvis_2021(_root)
    register_all_ytvis_2022(_root)
    register_all_ovis(_root)
    register_all_coco_video(_root)
    from . import vps
    from . import vss



    root="/home/zlni/zlni/data/fishtrack_ytvis"
    if os.path.exists(root):
        splits = ["train", "val"]
        for split in splits:
            name = f"fishtrack_ytvis_{split}"
            # JSON 路径: .../annotations/train.json
            json_file = os.path.join(root, "annotations", f"{split}.json")
            # 图片根路径: .../JPEGImages/train (因为 JSON 里的 file_names 是 video/xxx.jpg)
            image_root = os.path.join(root, "images", split)
        
            # 执行注册
            register_fishtrack_ytvis_instances(name, _get_fishtrack_instances_meta(), json_file, image_root)