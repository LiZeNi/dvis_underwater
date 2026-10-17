# For close-vocabulary
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus
export CUDA_VISIBLE_DEVICES=4,5
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 1 \
#   --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/configs/dvis_Plus/ovis/CTVIS_R50_fishtrack.yaml \
#   --dist-url tcp://127.0.0.1:50353 \
#   --resume MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/dvis++/CTVIS_R50_OVIS/model_final.pth \
#   #SOLVER.IMS_PER_BATCH 2

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/configs/dvis_Plus/ovis/DVIS_Plus_Online_R50_fishtrack.yaml \
  --dist-url tcp://127.0.0.1:50753 \
  SOLVER.IMS_PER_BATCH 2