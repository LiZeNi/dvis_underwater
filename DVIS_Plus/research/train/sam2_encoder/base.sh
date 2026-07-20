# For close-vocabulary
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus
export CUDA_VISIBLE_DEVICES=6,7

# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/sam2_encoder/base/CTVIS.yaml \
#   --dist-url tcp://127.0.0.1:50853 \
#   MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/CTVIS_SAM2_base_OVIS_untrain.pth \
#   SOLVER.IMS_PER_BATCH 2

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/sam2_encoder/base/DVIS_Plus_online.yaml\
  --dist-url tcp://127.0.0.1:50653 \
  --resume MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/research/DVIS++_Online_sam2_base/model_final.pth \
  SOLVER.IMS_PER_BATCH 2
