import sys
import torch

print("Python 路径：", sys.executable)
print("PyTorch 版本：", torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡：", torch.cuda.get_device_name(0))