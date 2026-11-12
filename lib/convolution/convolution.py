"""这里是手写提供的卷积库"""
import torch
from debug_system.debug_system import Debug_System
## 默认全核
DEFAULT_KERNEL_ALL: torch.tensor = torch.tensor(
    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]
).float().cuda()
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
## 标准四邻域拉普拉斯
KERNEL_LAPLACIAN = torch.tensor(
    [[ 0., -1.,  0.],
    [-1.,  4., -1.],
    [ 0., -1.,  0.]]
).float().cuda()
## 八邻域拉普拉斯（对斜向边缘更敏感）
KERNEL_LAPLACIAN_8 = torch.tensor(
    [[-1., -1., -1.],
    [-1.,  8., -1.],
    [-1., -1., -1.]]
).float().cuda()
## 高斯模糊 3 x 3
KERNEL_GAUSSIAN_3x3 = torch.tensor(
    [[1., 2., 1.],
    [2., 4., 2.],
    [1., 2., 1.]]
).float().cuda()
## 锐化核
KERNEL_SHARPEN = torch.tensor(
    [[ 0., -1.,  0.],
    [-1.,  5., -1.],
    [ 0., -1.,  0.]]
).float().cuda()

DEFAULT_SPRITE_SIZE: int = 3

## 核心Convolution类
class Convolution:
    """目前看起来卷积库并不需要什么初始化"""
    def __init__(self):
        pass

    """第一层卷积计算，初始化卷积模拟"""
    def get_convolution_layer_one(
            picture_mask: torch.tensor, 
            kernel: torch.tensor = DEFAULT_KERNEL_SOBEL_X
        ) -> torch.tensor:

        ## 使用默认 Kernerl核做运算,获得窗口集合
        matrix_windows_set: torch.tensor = picture_mask.unfold(
            dimension=0, 
            size=DEFAULT_SPRITE_SIZE, 
            step=1
            ).unfold(
                dimension=1, 
                size=DEFAULT_SPRITE_SIZE, 
                step=1
                )
        ## 展平矩阵计算第一层卷积
        matrix_windows_dimension_one: torch.tensor = matrix_windows_set.flatten(-2, -1)
        matrix_kernel_dimension_one: torch.tensor = kernel.flatten()
        ## 需要图像展平矩阵先乘，否则kernel展平矩阵无法乘到图像展平矩阵
        convolution_result: torch.tensor = matrix_windows_dimension_one @ matrix_kernel_dimension_one
        return convolution_result

    """第一层卷积计算，归一化处理(正向留存)带Padding"""
    def get_convolution_layer_one_normalization(
            picture_mask: torch.tensor, 
            kernel: torch.tensor = DEFAULT_KERNEL_SOBEL_X
        ) -> torch.tensor:
        ## 补充边缘(Padding)
        ker_height, ker_width = kernel.shape
        pad_top = ker_height // 2
        pad_bottom = ker_height // 2
        pad_left  = ker_height // 2
        pad_right = ker_height // 2

        ## padding 补充边缘
        picture_mask = torch.nn.functional.pad(
            picture_mask,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0
        )

        ## 使用默认 Kernerl核做运算,获得窗口集合
        matrix_windows_set: torch.tensor = picture_mask.unfold(
            dimension=0, 
            size=DEFAULT_SPRITE_SIZE, 
            step=1
            ).unfold(
                dimension=1, 
                size=DEFAULT_SPRITE_SIZE, 
                step=1
                )
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

    """第一层卷积，多通道(单图片)"""
    def get_convolution_layer_one_multi_channel(
            image: torch.tensor,
            kernels: torch.tensor = torch.stack(
                [torch.stack(
                    [DEFAULT_KERNEL_SOBEL_X,
                     DEFAULT_KERNEL_SOBEL_X,
                     DEFAULT_KERNEL_SOBEL_Y,
                     DEFAULT_KERNEL_ALL]
                )]
            )
        ):
        """
        多通道卷积
        @param image: 输入图像集合或者图像
        @param kernels: 卷积核集合 
        """
        ## 获得输入通道以及输出通道等基本信息
        img_height, img_width, channel_in = image.shape
        channel_out, ker_cha_in, ker_height, ker_width = kernels.shape
        ## 检查通道是否符合要求
        assert ker_cha_in == channel_in, "Kernel单核输入通道数量与图像输入通道(信号路数)不一致"
        assert ker_height == ker_width, "Kernel单核高度和宽度不一致"

        ## 拆分出每一个通道的单独的 tensor
        ## 最终图像结果
        result_image = torch.zeros_like(image)

        ## 拆分每一个通道，然后单独处理kernel核
        for channel in range(channel_in):
            ## 获取单独通道
            single_channel_image = image[..., channel]
            single_channel_image = single_channel_image
            ## 对单通道做一次卷积
            single_convolution_result = Convolution.get_convolution_layer_one_normalization(
                single_channel_image,
                kernel=kernels[0, channel]
            )
            ## 替换算出的通道
            single_convolution_result = single_convolution_result
            result_image[:, :, channel] = single_convolution_result

        Debug_System.debug_log("多通道卷积", f"多通道卷积结果: {result_image}")

        return result_image

    """第二层卷积计算，多通道"""
    def get_convolution_layer_second(picture_mask: torch.tensor):
        pass