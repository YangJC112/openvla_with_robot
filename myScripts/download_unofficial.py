import datasets
# ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "fractal20220817_data", streaming=True, split='train')  # IterDataset

#Optional subdatasets: 这里是表对应的 Registered Dataset Name ;从Dataset栏选择（对应Optional subdatasets (Full Name):），名字在Registered Dataset Name 
# fractal20220817_data 
# kuka
# bridge
# taco_play
# jaco_play
# berkeley_cable_routing
# roboturk
# nyu_door_opening_surprising_effectiveness
# viola
# berkeley_autolab_ur5
# toto
# language_table
# columbia_cairlab_pusht_real
# stanford_kuka_multimodal_dataset_converted_externally_to_rlds
# nyu_rot_dataset_converted_externally_to_rlds
# stanford_hydra_dataset_converted_externally_to_rlds
# austin_buds_dataset_converted_externally_to_rlds  ##############
# nyu_franka_play_dataset_converted_externally_to_rlds
# maniskill_dataset_converted_externally_to_rlds
# furniture_bench_dataset_converted_externally_to_rlds
# cmu_franka_exploration_dataset_converted_externally_to_rlds
# ucsd_kitchen_dataset_converted_externally_to_rlds
# ucsd_pick_and_place_dataset_converted_externally_to_rlds
# austin_sailor_dataset_converted_externally_to_rlds
# austin_sirius_dataset_converted_externally_to_rlds
# bc_z
# usc_cloth_sim_converted_externally_to_rlds
# utokyo_pr2_opening_fridge_converted_externally_to_rlds
# utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds
# utokyo_saytap_converted_externally_to_rlds
# utokyo_xarm_pick_and_place_converted_externally_to_rlds
# utokyo_xarm_bimanual_converted_externally_to_rlds
# robo_net
# berkeley_mvp_converted_externally_to_rlds
# berkeley_rpt_converted_externally_to_rlds
# kaist_nonprehensile_converted_externally_to_rlds
# stanford_mask_vit_converted_externally_to_rlds
# tokyo_u_lsmo_converted_externally_to_rlds
# dlr_sara_pour_converted_externally_to_rlds
# dlr_sara_grid_clamp_converted_externally_to_rlds
# dlr_edan_shared_control_converted_externally_to_rlds
# asu_table_top_converted_externally_to_rlds
# stanford_robocook_converted_externally_to_rlds
# eth_agent_affordances
# imperialcollege_sawyer_wrist_cam
# iamlab_cmu_pickup_insert_converted_externally_to_rlds
# uiuc_d3field
# utaustin_mutex
# berkeley_fanuc_manipulation
# cmu_playing_with_food
# cmu_play_fusion
# cmu_stretch
# berkeley_gnm_recon
# berkeley_gnm_cory_hall
# berkeley_gnm_sac_son



#Optional subdatasets (Full Name):
# RT-1 Robot Action 
# QT-Opt
# Berkeley Bridge
# Freiburg Franka Play
# USC Jaco Play
# Berkeley Cable Routing
# Roboturk
# NYU VINN
# Austin VIOLA
# Berkeley Autolab UR5
# TOTO Benchmark
# Language Table
# Columbia PushT Dataset
# Stanford Kuka Multimodal
# NYU ROT
# Stanford HYDRA
# Austin BUDS
# NYU Franka Play
# Maniskill
# Furniture Bench
# CMU Franka Exploration
# UCSD Kitchen
# UCSD Pick Place
# Austin Sailor
# Austin Sirius
# BC-Z
# USC Cloth Sim
# Tokyo PR2 Fridge Opening
# Tokyo PR2 Tabletop Manipulation
# Saytap
# UTokyo xArm PickPlace
# UTokyo xArm Bimanual
# Robonet
# Berkeley MVP Data
# Berkeley RPT Data
# KAIST Nonprehensile Objects
# QUT Dynamic Grasping
# Stanford MaskVIT Data
# LSMO Dataset
# DLR Sara Pour Dataset
# DLR Sara Grid Clamp Dataset
# DLR Wheelchair Shared Control
# ASU TableTop Manipulation
# Stanford Robocook
# ETH Agent Affordances
# Imperial Wrist Cam
# CMU Franka Pick-Insert Data
# QUT Dexterous Manpulation
# MPI Muscular Proprioception
# UIUC D3Field
# Austin Mutex
# Berkeley Fanuc Manipulation
# CMU Food Manipulation
# CMU Play Fusion
# CMU Stretch
# RECON
# CoryHall
# SACSoN
# RoboVQA
# ALOHA

# path = r"F:\td4\上海创智-人工智能联培\code\openvla\datasets_rlds"
# datasets_to_download = [
#     ("austin_buds_dataset_converted_externally_to_rlds", path),
#     ("cmu_franka_exploration_dataset_converted_externally_to_rlds", path),
#     ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", path),
# ]
# for subset_name, cache_path in datasets_to_download:
#     print(f"Downloading {subset_name} to {cache_path}...")
#     ds = datasets.load_dataset("jxu124/OpenX-Embodiment", subset_name, 
#                                streaming=True, split='train', cache_dir=cache_path)
#     print(f"Finished downloading {subset_name}.")

# 定义基础路径

# os.environ["HF_DATASETS_CACHE"] = "F:\td4\上海创智-人工智能联培\code\openvla\datasets_rlds"

# 定义数据集名称和子集名称
# datasets_to_download = [
#     ("austin_buds_dataset_converted_externally_to_rlds", "austin_buds"),
#     ("cmu_franka_exploration_dataset_converted_externally_to_rlds", "cmu_franka"),
#     ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", "utokyo_pr2"),
# ]
import os
import datasets

# 定义要下载的数据集列表
datasets_to_download = [
    ("austin_buds_dataset_converted_externally_to_rlds", "austin_buds"),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", "cmu_franka"),
    ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", "utokyo_pr2"),
]



ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "austin_buds_dataset_converted_externally_to_rlds", streaming=True, split='train') 