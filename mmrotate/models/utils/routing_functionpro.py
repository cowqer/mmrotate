import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import trunc_normal_

class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
# class RountingFunctionPro(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0, maximum=10):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        ############################################
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        ############################################
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        
        self.dropout3= nn.Dropout(dropout_rate)
        self.fc_aspect_ratio = nn.Linear(in_channels, kernel_number, bias=False)
        self.maximum = maximum

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)
        trunc_normal_(self.fc_aspect_ratio.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        ar = self.dropout2(x)
        ar = self.fc_aspect_ratio(ar)
        ar = torch.sigmoid(ar) * self.maximum
        
        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return ar, alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)
class RountingFunctionPro(nn.Module):
    def __init__(self, in_channels, kernel_number=1):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 新增的 fc 层，用于预测卷积核的宽高
        self.fc_width = nn.Linear(in_channels, kernel_number, bias=True)
        self.fc_height = nn.Linear(in_channels, kernel_number, bias=True)
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)

        trunc_normal_(self.fc_width.weight, std=.02)
        trunc_normal_(self.fc_height.weight, std=.02)

    def forward(self, x):
        # 卷积 + 归一化 + 激活
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        # 平均池化
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        # 计算 alphas 和 angles

        # 计算卷积核的宽和高
        width = self.fc_width(x)  # [batch_size, kernel_number]
        height = self.fc_height(x)  # [batch_size, kernel_number]

        # 限制宽度和高度在 3 到 7 之间
        width = torch.clamp(width, min=3, max=7)
        height = torch.clamp(height, min=3, max=7)

        # 将输出转换为整数
        width = torch.round(width).long()  # 或者使用 torch.floor(width).long() 或 torch.ceil(width).long()
        height = torch.round(height).long()  # 同理

        return  width, height
