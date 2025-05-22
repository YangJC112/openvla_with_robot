#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# # 激活 Conda 环境
# conda init
# conda activate openvla
# tar -czvf openvla_org_inscripts.tar.gz /root/autodl-tmp/openvla/myScripts/autodl-tmp
# 运行微调脚本
torchrun --standalone --nnodes 1 --nproc-per-node 8 /home/chuangzhi/zhq/yjc/openvla/vla-scripts/finetune.py \
  --vla_path "/home/chuangzhi/zhq/yjc/openvla7b_huggingfacemodel" \
  --data_root_dir '/home/chuangzhi/zhq/yjc/modified_libero_rlds' \
  --dataset_name libero_spatial_no_noops \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 15000\
  --max_steps 30000  \
  --wandb_project 123 \
  --wandb_entity 1275259847-tianjin-university

torchrun --standalone --nnodes 1 --nproc-per-node 8 /home/chuangzhi/zhq/yjc/openvla/vla-scripts/finetune.py \
  --vla_path "/home/chuangzhi/zhq/yjc/runs/openvla7b_huggingfacemodel+libero_spatial_no_noops+b2+lr-0.0005+lora-r32+dropout-0.0" \
  --data_root_dir '/home/chuangzhi/zhq/yjc/modified_libero_rlds' \
  --dataset_name libero_spatial_no_noops \
  --batch_size 1 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 15000\
  --max_steps 30000  \
  --wandb_project 123 \
  --wandb_entity 1275259847-tianjin-university
  # --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  # --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  # --lora_rank 32  \
  # --wandb_project <PROJECT> \
  # --wandb_entity <ENTITY> \