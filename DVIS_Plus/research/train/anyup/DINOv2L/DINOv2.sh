# For close-vocabulary
cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus
export PYTHONPATH=$PYTHONPATH:/data/xyjiang/zlni/zlni/anyup
export CUDA_VISIBLE_DEVICES=0,1
# train the segmenter of the DVIS++
# python train_net_video.py \
#   --num-gpus 2 \
#   --config-file  /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/anyup/DINOv2/CTVIS.yaml\
#   --dist-url tcp://127.0.0.1:50053 \
#   MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_pretrain_weight/dvis++/ctvis_vitl_ovis.pth\
#   SOLVER.IMS_PER_BATCH 2

# # train the DVIS++(online)
python train_net_video.py \
  --num-gpus 2 \
  --config-file  /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_Plus/research/configs/anyup/DINOv2/DVIS_Plus_Online.yaml\
  --dist-url tcp://127.0.0.1:50853 \
  --resume MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/research/anyup/DINOv2/DVIS_Plus_Online/model_0011999.pth \
  SOLVER.IMS_PER_BATCH 2