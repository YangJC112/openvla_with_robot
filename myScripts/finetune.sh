#!/bin/bash

# # 激活 Conda 环境
# conda init
# conda activate openvla
# tar -czvf openvla_org_inscripts.tar.gz /root/autodl-tmp/openvla/myScripts/autodl-tmp
# 运行微调脚本
torchrun --standalone --nnodes 1 --nproc-per-node 1 /root/autodl-tmp/openvla/vla-scripts/finetune.py \
  --vla_path "/root/autodl-tmp/openvla/myScripts/autodl-tmp/huggingface_models" \
  --data_root_dir '/root/autodl-tmp/tensorflow_datasets' \
  --dataset_name austin_buds_dataset_converted_externally_to_rlds \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 1000\
  --max_steps 20000  
  # --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  # --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  # --lora_rank 32  \
  # --wandb_project <PROJECT> \
  # --wandb_entity <ENTITY> \