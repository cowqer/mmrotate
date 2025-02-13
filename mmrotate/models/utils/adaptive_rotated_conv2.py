import torch
import torch.nn as nn
from torch.nn import functional as F
from .adaptive_rotated_conv import AdaptiveRotatedConv2d, batch_rotate_multiweight
from .Gatedpconv import GatedPConv

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AdaptiveRotatedConv2d1(AdaptiveRotatedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias,
                         kernel_number, rounting_func, rotate_func)
        
        # 添加新的属性或方法
        self.gatedpc = GatedPConv(out_channels, out_channels, k=self.kernel_size, s=1)
        self.stn = SpatialAttention(7)
        nn.init.kaiming_normal_(self.additional_layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 调用父类的 forward 方法
        out1 = self.gatedpc(x)
        
        out2 = super().forward(x)
        
        out3 = self.stn(x) 
        
        out = out1 + out2
        
        out = out * out3
          
        return out

    def extra_repr(self):
        s = super().extra_repr()
        s += ', additional_layer=True'
        return s

