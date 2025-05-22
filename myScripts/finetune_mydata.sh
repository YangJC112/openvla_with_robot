#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # 激活 Conda 环境
# conda init
# conda activate openvla
# tar -czvf openvla_org_inscripts.tar.gz /root/autodl-tmp/openvla/myScripts/autodl-tmp
# 运行微调脚本
torchrun --standalone --nnodes 1 --nproc-per-node 8 /home/chuangzhi/zhq/yjc/openvla/vla-scripts/finetune.py \
  --vla_path "/home/chuangzhi/zhq/yjc/runs/openvla7b_huggingfacemodel+libero_spatial_no_noops+b2+lr-0.0005+lora-r32+dropout-0.0" \
  --data_root_dir '/home/chuangzhi/zhq/yjc/mydata_tensorflow_datasets' \
  --dataset_name example_dataset \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 5000\
  --max_steps 5000  \
  --wandb_project 123 \
  --wandb_entity 1275259847-tianjin-university