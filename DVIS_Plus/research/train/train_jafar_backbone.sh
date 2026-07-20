# For close-vocabulary
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus
export CUDA_VISIBLE_DEVICES=4,5,6,7

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 4 \
  --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/jafar_backbone.yaml\
  --dist-url tcp://127.0.0.1:50653 \
  --resume MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/research/dvis++/jafar_backbone/model_0015999.pth \
  SOLVER.IMS_PER_BATCH 4