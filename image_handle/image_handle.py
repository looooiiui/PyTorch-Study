"""
这里是提供给模型训练的图形工具库
主要提供读取的图像的矩阵，灰度图等
"""
import os
import numpy
import torch
import numpy
from PIL import Image
from pathlib import Path

## 当前python绝对路径
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPT_DIR.parent
## 文件默认读取路径
DEFAULT_PICTURE_PATH: str = os.path.join(PROJECT_ROOT, "train_image")
## 默认归一化:硬数值归一值
DEFAULT_NORMALIZATION: int = 255

class Image_Handle:
    """目前看起来图形库好像不需要什么属性"""
    def __init__(self):
        pass

    """返回读取文件灰度图矩阵"""
    def load_training_gray_image(file_name: str, device: torch.device) -> torch.tensor:
        image = Image.open(os.path.join(DEFAULT_PICTURE_PATH, file_name)).convert("L")
        ## 灰度图先转换为数组才能给torch.tensor处理
        image_numpy = numpy.array(image)
        image_tensor = torch.tensor(image_numpy, dtype=torch.float32)/DEFAULT_NORMALIZATION
        ## 搬运数据到设备上
        image_tensor = image_tensor.to(device)
        return image_tensor

    """将矩阵图像重新处理回图像，用于可视化训练结果"""
    def convert_tensor_to_image(in_tensor: torch.tensor):
        in_tensor = in_tensor.cpu()
        image_output = Image.fromarray((in_tensor.numpy() * 255).astype("uint8"))
        image_output.save("feature_out.png")