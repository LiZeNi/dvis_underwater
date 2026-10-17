# For close-vocabulary
cd /data/zlni/zlni/DVIS_Plus/DVIS_Plus
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/data/zlni/zlni
export CUDA_VISIBLE_DEVICES=0,1
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/depth/cen/CTVIS.yaml\
#   --dist-url tcp://127.0.0.1:51053 \
#   --resume MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/research/depth/cen/CTVIS_DIONv2l/model_0013999.pth\
#   SOLVER.IMS_PER_BATCH 2

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/depth/cen/dvis++.yaml \
  --dist-url tcp://127.0.0.1:51253 \
  MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/research/depth/cen/CTVIS_DIONv2l/model_final.pth \
  SOLVER.IMS_PER_BATCH 2