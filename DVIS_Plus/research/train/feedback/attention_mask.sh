# For close-vocabulary
cd /data/zlni/zlni/DVIS_Plus/DVIS_Plus
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/data/zlni/zlni/anyup
export CUDA_VISIBLE_DEVICES=0,1
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/feedback/attention_mask/CTVIS.yaml \
#   --dist-url tcp://127.0.0.1:50173 \
#   --resume MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/dvis++/feedback/attention_mask/DINO_vitl/CTVIS_DINO2_OVIS/model_0015999.pth\
#   SOLVER.IMS_PER_BATCH 2

# # # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/feedback/attention_mask/dvis++.yaml\
  --dist-url tcp://127.0.0.1:50953 \
  --resume MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/dvis++/feedback/attention_mask/DINO_vitl/DVIS_Plus_Online_DINO2_OVIS/model_final.pth \
  SOLVER.IMS_PER_BATCH 2