cd /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_DAQ
export CUDA_VISIBLE_DEVICES=2,3
# train the DVIS-DAQ(online), take ovis dataset as example
python train_net_video.py \
  --num-gpus 1 \
  --config-file /data/xyjiang/zlni/zlni/DVIS_Plus/DVIS_DAQ/configs/dvis_daq/ovis/DAQ_Online_R50_fishtrack.yaml \
  --dist-url tcp://127.0.0.1:50253 \
  --resume MODEL.WEIGHTS /data/xyjiang/zlni/zlni/data/UMOT_weight/dvis_daq/DVIS_Plus_Online_R50_OVIS/model_final.pth \
  SOLVER.IMS_PER_BATCH 1\
  MODEL.MASK_FORMER.TEST.MAX_NUM 10