#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
# # 激活 Conda 环境
# conda init
# conda activate openvla
# tar -czvf openvla_org_inscripts.tar.gz /root/autodl-tmp/openvla/myScripts/autodl-tmp
# 运行微调脚本
accelerate launch --num_processes 1  --mixed_precision=bf16 /home/chuangzhi/zhq/yjc/openvla/vla-scripts/finetune2.py \
  --vla_path "/home/chuangzhi/zhq/yjc/openvla7b_huggingfacemodel" \
  --data_root_dir '/home/chuangzhi/zhq/yjc/modified_libero_rlds' \
  --dataset_name libero_spatial_no_noops \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 1000\
  --max_steps 20000  \
  --wandb_project 123 \
  --wandb_entity 1275259847-tianjin-university
  # --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  # --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  # --lora_rank 32  \
  # --wandb_project <PROJECT> \
  # --wandb_entity <ENTITY> \