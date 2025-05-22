import os
from transformers import AutoProcessor

# 设置环境变量，指定 Hugging Face 镜像站点
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 指定缓存目录
cache_dir = "/home/chuangzhi/zhq/yjc/openvla-7b-finetuned-libero-spatial_huggingfacemodel"

# 下载并加载模型
processor = AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial", cache_dir=cache_dir, trust_remote_code=True)

print(f"Model files are downloaded to: {cache_dir}")


'''使用下面的命令
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download openvla/openvla-7b-finetuned-libero-spatial  --local-dir openvla-7b-finetuned-libero-spatia
l_huggingfacemodel
'''