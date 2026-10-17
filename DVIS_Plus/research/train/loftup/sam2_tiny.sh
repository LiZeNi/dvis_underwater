# For close-vocabulary
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus
export CUDA_VISIBLE_DEVICES=4,5

# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file  /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/loftup/sam2_tiny/CTVIS_sam2_tiny.yaml\
#   --dist-url tcp://127.0.0.1:50353 \
#   MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/CTVIS_SAM2_Tiny_OVIS_untrain.pth \
#   SOLVER.IMS_PER_BATCH 2
#/data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/CTVIS_SAM2_Tiny_OVIS_untrain.pth

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/loftup/sam2_tiny/DVIS_Plus_online_sam2_tiny.yaml\
  --dist-url tcp://127.0.0.1:50553 \
  MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/research/loftup/CTVIS_Online_sam2_tiny/model_final.pth \
  SOLVER.IMS_PER_BATCH 2

