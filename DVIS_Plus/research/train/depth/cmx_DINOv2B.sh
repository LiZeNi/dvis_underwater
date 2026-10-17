# For close-vocabulary
cd /data/zlni/zlni/DVIS_Plus/DVIS_Plus
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/data/zlni/zlni
export CUDA_VISIBLE_DEVICES=4,5
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/depth/cmx/CTVIS.yaml\
#   --dist-url tcp://127.0.0.1:51053 \
#   --resume MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/research/depth/cmx/CTVIS_DIONv2B/model_final.pth\
#   SOLVER.IMS_PER_BATCH 2

# train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/depth/cmx/dvis++.yaml \
  --dist-url tcp://127.0.0.1:51953 \
  --resume MODEL.WEIGHTS  /data/zlni/zlni/data/UMOT_weight/research/depth/cmx/dvis++_DIONv2l/model_final.pth\
  SOLVER.IMS_PER_BATCH 2

