@echo off
REM 激活 Conda 环境
call conda activate openvla

REM 运行微调脚本
python -m vla-scripts.finetune ^
  --vla_path "C:\Users\YangJC\.cache\huggingface\hub\models--openvla--openvla-7b\snapshots\openvla" ^
  --data_root_dir "C:\Users\YangJC\Desktop\tensorflow_datasets" ^
  --dataset_name austin_buds_dataset_converted_externally_to_rlds ^
  --run_root_dir "F:/td4/Sii/code/openvla/logs" ^
  --adapter_tmp_dir "F:/td4/Sii/code/openvla/tmp" ^
  --lora_rank 32 ^
  --batch_size 16 ^
  --grad_accumulation_steps 1 ^
  --learning_rate 5e-4 ^
  --image_aug True ^
  @REM --wandb_project "your_project_name" ^
  @REM --wandb_entity "your_wandb_entity" ^
  --save_steps 100