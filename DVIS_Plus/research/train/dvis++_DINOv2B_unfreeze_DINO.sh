# For close-vocabulary
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus
export CUDA_VISIBLE_DEVICES=0,1
# train the segmenter of the DVIS++
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/configs/dvis_Plus/ovis/vit_adapter/CTVIS_VitAdapterB_unfreeze.yaml\
  --dist-url tcp://127.0.0.1:50053 \
  MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_vitl_ovis.pth\
  SOLVER.IMS_PER_BATCH 2

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/configs/dvis_Plus/ovis/vit_adapter/DVIS_Plus_Online_VitAdapterB_unfreeze.yaml\
  --dist-url tcp://127.0.0.1:50853 \
  MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/dvis++/DINO_vitb/unfreeze_DINO/CTVIS_DINO2_OVIS/model_final.pth \
  SOLVER.IMS_PER_BATCH 2