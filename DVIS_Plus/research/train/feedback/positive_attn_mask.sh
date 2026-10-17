# For close-vocabulary
cd /data/zlni/zlni/DVIS_Plus/DVIS_Plus
export PYTHONPATH=$PYTHONPATH:/data/zlni/zlni/anyup
export CUDA_VISIBLE_DEVICES=0,1
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/feedback/positive_attn_mask/CTVIS.yaml\
#   --dist-url tcp://127.0.0.1:50053 \
#   MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_vitl_ovis.pth\
#   SOLVER.IMS_PER_BATCH 2

# train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/feedback/positive_attn_mask/dvis++.yaml\
  --dist-url tcp://127.0.0.1:59853 \
  --resume MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/dvis++/feedback/positive_attn_mask/DINO_vitl/DVIS_Plus_Online_DINO2_OVIS_seg_freeze_stage2/model_0019999.pth\
  SOLVER.IMS_PER_BATCH 2