# For close-vocabulary
cd /data/zlni/zlni/DVIS_Plus/DVIS_Plus
export PYTHONPATH=$PYTHONPATH:/data/zlni/zlni/anyup
export CUDA_VISIBLE_DEVICES=2,3
# train the segmenter of the DVIS++
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/feedback/concat/CTVIS.yaml\
  --dist-url tcp://127.0.0.1:50173 \
  MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_vitl_ovis.pth\
  SOLVER.IMS_PER_BATCH 2

# # # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/feedback/concat/DVIS_Plus_Online.yaml\
  --dist-url tcp://127.0.0.1:50953 \
  MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/dvis++/feedback/concat/CTVIS_DINO2_OVIS/model_final.pth \
  SOLVER.IMS_PER_BATCH 2