import torch
import torch.nn as nn
from torch.nn import functional as F
from .adaptive_rotated_conv import AdaptiveRotatedConv2d, batch_rotate_multiweight
from .Gatedpconv import GatedPConv

class AdaptiveRotatedConv2d1(AdaptiveRotatedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias,
                         kernel_number, rounting_func, rotate_func)
        
        # 添加新的属性或方法
        self.gatedpc = GatedPConv(self.out_channels, self.out_channels, k=3, s=1)
        

    def forward(self, x):
        # 调用父类的 forward 方法
        out1 = self.gatedpc(x)
        
        out2 = super().forward(x)
        
        out = out1 * out2
        # out = out1 + out2
        
        
        return out



