'''
测试 PyTorch 的GPU使用
用于测试机器学习的脚本
'''
import torch
import PIL

## 自定义库
from image_handle.image_handle import Image_Handle
from image_handle.image_handle import Image_Kind
from lib.convolution.convolution import Convolution
from debug_system.debug_system import Debug_System

## 结果+其余模拟变量
matrix_cal_result: torch.tensor

## 程序入口，目前程序较小，未被使用
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_image: torch.tensor = Image_Handle.load_training_gray_image("train_image.png", device, Image_Kind.RGBNormal)

    ## 测试区
    result_feature: torch.tensor = Convolution.get_convolution_layer_one_multi_channel(train_image)
    Image_Handle.convert_tensor_to_image(result_feature)
    Debug_System.debug_log("主控", "结束程序")
