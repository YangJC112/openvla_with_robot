export CUDA_VISIBLE_DEVICES=6,7
python /home/chuangzhi/zhq/yjc/openvla/experiments/robot/libero/run_libero_eval.py   \
--model_family openvla   \
--pretrained_checkpoint /home/chuangzhi/zhq/yjc/runs/openvla7b_huggingfacemodel+libero_spatial_no_noops+b2+lr-0.0005+lora-r32+dropout-0.0    \
--task_suite_name libero_spatial   \
--center_crop True  \
--num_trials_per_task 10



            # "args": [
            #     "--model_family", "openvla",
            #     "--pretrained_checkpoint", "/home/chuangzhi/zhq/yjc/runs/4-3+b2+lr-5e-05+lora-r32+dropout-0.0",
            #     "--task_suite_name", "libero_spatial",
            #     "--center_crop", "True"
            # ],
            # "env": {
            #     "CUDA_VISIBLE_DEVICES": "6,7"
            # }