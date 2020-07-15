'''
fcn.py 的任务
1. 实现双线插值 
2. 实现FCN本人  
(人 •͈ᴗ•͈)  ۶♡♡
'''

import numpy as numpy
import torch
from torch import nn
from torchvision import models

#双插
def Bilinear_interpolation (src, new_size):
    '''
    使用双线性插值方法放大图像
    params：
        src(np.ndarray)：输入图片
        new_size(tuple)：目标尺寸
    ret：
        dst(np.ndarry):目标图像
    '''
    dst_h, dst_w = new_size  #目标图像的hw
    src_h, src_w = src.shape[:2] #原始图像的hw

    #如果跟需求符合， 就不需要缩放，直接拷贝
    if src_h == dst_h and src_w == dst_w:
        return src.copy()

    scale_x = float(src_w) / dst_w
    scale_y = float(src_H) / dst_h

    #遍历目标图上的每个像素点
    ##构建一张目标大小的空图，遍历差值
    dst = np.zeros((dst_h,dst_w,3),dtype=np.int8)
    ##因为是彩色图，遍历三层： a.rgb三通道 b.height c.width
    for n in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #目标像素在原图上的坐标 src+0.5 = (dst_x + 0.5) *scale_x
                #加0.5的偏差，可以保证图像缩小时，不会漏掉像素点 详细看：https://www.cnblogs.com/kk17/p/9989984.html
            src_x = (dst_x + 0.5)*scale_x -0.5
            src_y = (dst_y + 0.5)*scale_y -0.5

            #计算在原图某像素点的4个近邻点的位置
            src_x_0 = int(np.floor(src_x)) #*floor()向下取整数 ex: floor(1.2) = 1.0
            src_y_0 = int(np.floor(src_y))
            src_x_1 = min(src_x_0 + 1, src_w - 1  ) #防止出界
            src_y_1 = min(src_y_0 + 1, src_h - 1  )


'''
初始化反卷积核
'''
def bilinear_kernel(in_channels, out_channels, kernel_size):


    return

pretrained_net = models.vgg16_bn(pretrained=False)

#FCN本人
class FCN(nn.Module):
    def __init__(self, num_classes):

    def forward(self, x):

        return


if __name__ == "__main__":
    rgb = torch.randn(1，3，352，480)
    net = FCN(12)
    out = net(rgb)
    print('喵喵喵喵喵喵喵喵---------------')
    print(out.shape)