import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Current device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

import tensorflow as tf
print(tf.__version__)