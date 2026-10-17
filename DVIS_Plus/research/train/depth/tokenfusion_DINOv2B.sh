# For close-vocabulary
cd /data/zlni/zlni/DVIS_Plus/DVIS_Plus
export PYTHONPATH=$PYTHONPATH:/data/zlni/zlni
export CUDA_VISIBLE_DEVICES=0,1
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file  /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/depth/tokenfusion/CTVIS.yaml\
#   --dist-url tcp://127.0.0.1:50053 \
#   MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_vitl_ovis.pth\
#   SOLVER.IMS_PER_BATCH 2

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file /data/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/depth/tokenfusion/dvis++.yaml \
  --dist-url tcp://127.0.0.1:50853 \
  MODEL.WEIGHTS /data/zlni/zlni/data/UMOT_weight/research/depth/tokenfusion/CTVIS_DIONv2L/model_final.pth \
  SOLVER.IMS_PER_BATCH 2