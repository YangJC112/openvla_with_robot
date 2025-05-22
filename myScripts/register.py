import tensorflow_datasets as tfds
# import os
import sys
sys.path.append('/home/chuangzhi/zhq/yjc/rlds_dataset_builder')
from example_dataset.example_dataset_dataset_builder import ExampleDataset
# # 添加项目根目录到 Python 路径
sys.path.append('/home/chuangzhi/zhq/yjc/rlds_dataset_builder')
# # # 注册数据集目录
# # tfds.core.registered.register_data_dir('/home/chuangzhi/zhq/yjc/rlds_dataset_builder/my_offline_datasets')
# import tensorflow_datasets as tfds
# from example_dataset.example_dataset_dataset_builder import ExampleDataset

# # 打印所有已注册的数据集（应包含你的数据集）
# print("所有数据集:", tfds.list_builders())

# # 尝试直接通过 builder 加载
# builder = ExampleDataset(data_dir="/home/chuangzhi/zhq/yjc/rlds_dataset_builder/my_offline_datasets")
# ds = builder.as_dataset(split="train")
# print("手动加载成功:", ds)
# # 直接加载数据集
# ds = tfds.load('example_dataset', split='train')
# # 指定自定义数据路径
custom_data_dir = '/home/chuangzhi/zhq/yjc/rlds_dataset_builder/my_offline_datasets'

# # 加载数据集
# builder = tfds.builder('example_dataset', data_dir=custom_data_dir)
# ds = builder.as_dataset(split='train')

dataset = tfds.load("example_dataset", data_dir=custom_data_dir, split="train")
print(dataset)