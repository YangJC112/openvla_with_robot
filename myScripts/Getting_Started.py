# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
#! export HF_ENDPOINT=https://hf-mirror.com
#! huggingface-cli download --resume-download openvla/openvla-7b --local-dir openvla7b_huggingfacemodel
# model_path = r"/root/autodl-tmp/huggingface_models/models--openvla--openvla-7b/snapshots/openvla_org"
model_path = r"/home/chuangzhi/zhq/yjc/openvla7b_huggingfacemodel"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path, 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
# image: Image.Image = get_from_camera(...)

# Grab image input & format prompt
image_path = r'myScripts/1.jpg' # 替换为你的图像路径
image = Image.open(image_path)  # 加载图像

prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(action)
# # Execute...
# robot.act(action, ...)