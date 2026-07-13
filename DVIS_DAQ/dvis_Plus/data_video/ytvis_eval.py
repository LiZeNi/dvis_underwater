# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from .datasets.ytvis_api.ytvos import YTVOS
from .datasets.ytvis_api.ytvoseval import YTVOSeval
from .datasets.pycocotools.oviseval import OVISeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table


class YTVISEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        # ===== 🔧 修改：支持分散的 JSON 文件 =====
        json_file = self._metadata.get("json_file", None)
        
        if json_file is None or not os.path.exists(PathManager.get_local_path(json_file)):
            # 如果没有合并的 JSON 文件，从分散的 JSON 创建
            self._logger.info(f"📝 No merged JSON file found, loading from scattered annotations...")
            self._ytvis_api = self._load_scattered_annotations()
        else:
            # 使用合并的 JSON 文件
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                self._ytvis_api = YTVOS(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._ytvis_api.dataset

    def _load_scattered_annotations(self):
        """
        从分散的 JSON 文件加载标注并创建 YTVOS 对象
        (最终修正版：支持损坏/空 JSON，使用占位符保证 ID 绝对对齐)
        """
        annotation_dir = self._metadata.get("annotation_dir", None)
        
        if annotation_dir is None:
            raise ValueError("Metadata must contain 'annotation_dir'.")
        
        annotation_dir = PathManager.get_local_path(annotation_dir)
        # 必须排序，确保和 DataLoader 读取顺序一致
        json_files = sorted(Path(annotation_dir).glob("*.json"))
        
        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {annotation_dir}")
        
        self._logger.info(f"🔄 Loading {len(json_files)} scattered JSON files from {annotation_dir}")
        
        # 获取类别信息 (逻辑不变)
        categories = self._metadata.get("thing_classes", None)
        if categories is None:
            # 尝试从第一个文件获取，如果失败则使用默认
            try:
                with open(json_files[0], 'r') as f:
                    sample_data = json.load(f)
                categories = sample_data.get("categories", [{"id": 1, "name": "object"}])
            except:
                categories = [{"id": 1, "name": "object"}]
        else:
            categories = [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(categories)]
        
        merged_data = {
            "info": {"description": "Merged Dataset", "version": "1.0"},
            "categories": categories,
            "videos": [],
            "annotations": []
        }
        
        video_id = 0
        ann_id = 0
        
        for json_file in json_files:
            video_name = json_file.stem
            
            # 1. 尝试读取文件
            coco_data = {}
            load_success = False
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                load_success = True
            except Exception as e:
                self._logger.warning(f"⚠️  File broken {json_file}: {e}. Using dummy placeholder.")
                load_success = False

            # 2. 尝试解析图像信息
            has_valid_images = False
            if load_success and "images" in coco_data and isinstance(coco_data["images"], list) and len(coco_data["images"]) > 0:
                has_valid_images = True

            # 3. 构建视频信息 (关键步骤)
            if has_valid_images:
                # === 情况 A: 有效文件 ===
                images_dict = {img["id"]: img for img in coco_data["images"]}
                sorted_images = sorted(images_dict.items(), key=lambda x: x[1]["file_name"])
                first_img = sorted_images[0][1]
                
                width = first_img.get("width", 1920) # 防止 width 缺失
                height = first_img.get("height", 1080)
                length = len(sorted_images)
                
                # 准备标注处理所需的映射
                frame_idx_map = {img_id: idx for idx, (img_id, _) in enumerate(sorted_images)}
                
            else:
                # === 情况 B: 无效/空/无图文件 ===
                # 创建一个“假”视频信息进行占位
                # 只要 ID 占住了，Evaluator 就不会报错，只是这个视频没有 GT 也就是了
                self._logger.info(f"⚠️  {video_name} has no images/invalid. Creating dummy entry.")
                width = 100  # 随意设置，不影响
                height = 100
                length = 0   # 长度为 0
                frame_idx_map = {} # 空映射

            # 无论有效无效，都要注册视频！
            video_record = {
                "id": video_id,  # 【核心】保持 ID 连续自增
                "name": video_name,
                "width": width,
                "height": height,
                "length": length
            }
            merged_data["videos"].append(video_record)

            # 4. 处理标注 (仅当文件有效且有标注时)
            if has_valid_images and "annotations" in coco_data and len(coco_data["annotations"]) > 0:
                track_anns = defaultdict(lambda: {
                    "segmentations": [None] * length,
                    "bboxes": [None] * length,
                    "areas": [None] * length,
                    "category_id": None
                })
                
                for ann in coco_data["annotations"]:
                    # 鲁棒性获取字段
                    track_id = ann.get("track_id", ann.get("id"))
                    img_id = ann.get("image_id")
                    
                    if img_id not in frame_idx_map:
                        continue
                        
                    frame_idx = frame_idx_map[img_id]
                    
                    # 填充数据
                    segm = ann.get("segmentation")
                    if segm:
                        if isinstance(segm, dict):
                            track_anns[track_id]["segmentations"][frame_idx] = segm
                        elif isinstance(segm, list) and len(segm) > 0:
                            track_anns[track_id]["segmentations"][frame_idx] = segm
                    
                    bbox = ann.get("bbox")
                    if bbox and len(bbox) == 4:
                        track_anns[track_id]["bboxes"][frame_idx] = bbox
                    
                    area = ann.get("area", 0)
                    track_anns[track_id]["areas"][frame_idx] = area
                    
                    if track_anns[track_id]["category_id"] is None:
                        track_anns[track_id]["category_id"] = ann.get("category_id", 1)
                
                # 写入标注
                for track_id, track_data in track_anns.items():
                    ann_record = {
                        "id": ann_id,
                        "video_id": video_id,
                        "category_id": track_data["category_id"],
                        "segmentations": track_data["segmentations"],
                        "bboxes": track_data["bboxes"],
                        "areas": track_data["areas"],
                        "iscrowd": 0
                    }
                    merged_data["annotations"].append(ann_record)
                    ann_id += 1

            # 5. 【核心】ID 必须自增
            video_id += 1
            
        self._logger.info(f"✅ Total: {len(merged_data['videos'])} videos (including dummies), {len(merged_data['annotations'])} annotations")
        
        # ... (后续生成临时文件代码不变) ...
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(merged_data, f)
            temp_json = f.name
            
        with contextlib.redirect_stdout(io.StringIO()):
            ytvis_api = YTVOS(temp_json)
            
        try:
            os.unlink(temp_json)
        except:
            pass
            
        return ytvis_api


    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_coco_json_video(inputs, outputs)
        self._predictions.extend(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for YTVIS format ...")

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in predictions:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        coco_eval = (
            _evaluate_predictions_on_coco(
                self._ytvis_api,
                predictions,
            )
            if len(predictions) > 0
            else None  # cocoapi does not handle empty results very well
        )

        res = self._derive_coco_results(
            coco_eval, class_names=self._metadata.get("thing_classes")
        )
        self._results["segm"] = res

    def _derive_coco_results(self, coco_eval, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format("segm") + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format("segm") + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]
    video_length = inputs[0]["length"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]

    ytvis_results = []
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, masks)):
        segms = [
            mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in m
        ]
        for rle in segms:
            rle["counts"] = rle["counts"].decode("utf-8")

        res = {
            "video_id": video_id,
            "score": s,
            "category_id": l,
            "segmentations": segms,
        }
        ytvis_results.append(res)

    return ytvis_results


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    img_ids=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_results = copy.deepcopy(coco_results)
    # When evaluating mask AP, if the results contain bbox, cocoapi will
    # use the box area as the area of the instance, instead of the mask area.
    # This leads to a different definition of small/medium/large.
    # We remove the bbox field to let mask AP use mask area.
    for c in coco_results:
        c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    # coco_eval = YTVOSeval(coco_gt, coco_dt)
    coco_eval = OVISeval(coco_gt, coco_dt)
    # For COCO, the default max_dets_per_image is [1, 10, 100].
    max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
