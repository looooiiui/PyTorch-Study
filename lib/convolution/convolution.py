"""这里是手写提供的卷积库"""
import torch

## Kernel 核(sobel_x): 默认值模拟值
DEFAULT_KERNEL_SOBEL_X: torch.tensor = torch.tensor(
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
).float().cuda()
## Kernel 核(sobel_y): 默认值模拟值
DEFAULT_KERNEL_SOBEL_Y: torch.tensor = torch.tensor(
    [[-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]]
).float().cuda()

DEFAULT_SPRITE_SIZE: int = 3

## 核心Convolution类
class Convolution:
    """目前看起来卷积库并不需要什么初始化"""
    def __init__(self):

        pass
    """第一层卷积计算，初始化卷积模拟"""
    def get_convolution_layer_one(picture_mask: torch.tensor, kernel: torch.tensor = DEFAULT_KERNEL_SOBEL_X) -> torch.tensor:
        ## 使用默认 Kernerl核做运算,获得窗口集合
        matrix_windows_set: torch.tensor = picture_mask.unfold(dimension=0, size=DEFAULT_SPRITE_SIZE, step=1).unfold(dimension=1, size=DEFAULT_SPRITE_SIZE, step=1)
        ## 展平矩阵计算第一层卷积
        matrix_windows_dimension_one: torch.tensor = matrix_windows_set.flatten(-2, -1)
        matrix_kernel_dimension_one: torch.tensor = kernel.flatten()
        ## 需要图像展平矩阵先乘，否则kernel展平矩阵无法乘到图像展平矩阵
        convolution_result: torch.tensor = matrix_windows_dimension_one @ matrix_kernel_dimension_one
        return convolution_result

    """第一层卷积计算，归一化处理(正向留存)"""
    def get_convolution_layer_one_normalization(picture_mask: torch.tensor, kernel: torch.tensor = DEFAULT_KERNEL_SOBEL_Y) -> torch.tensor:
        ## 使用默认 Kernerl核做运算,获得窗口集合
        matrix_windows_set: torch.tensor = picture_mask.unfold(dimension=0, size=DEFAULT_SPRITE_SIZE, step=1).unfold(dimension=1, size=DEFAULT_SPRITE_SIZE, step=1)
        ## 展平矩阵计算第一层卷积
        matrix_windows_dimension_one: torch.tensor = matrix_windows_set.flatten(-2, -1)
        matrix_kernel_dimension_one: torch.tensor = kernel.flatten()
        ## 需要图像展平矩阵先乘，否则kernel展平矩阵无法乘到图像展平矩阵
        convolution_result: torch.tensor = matrix_windows_dimension_one @ matrix_kernel_dimension_one
        ## 归一化处理卷积
        result_min_num = convolution_result.min()
        result_max_num = convolution_result.max()
        normalization_matrix: torch.tensor = torch.clamp(convolution_result, min=0.0)
        normalization_matrix = (convolution_result - result_min_num) / (result_max_num - result_min_num)
        ## 返回归一处理后的矩阵
        return normalization_matrix