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
    



class new_RoutingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=90.0, proportion_alpha=1.0):
        super().__init__()
        self.kernel_number = kernel_number
        
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        self.proportion_alpha = proportion_alpha

        
        self.discrete_angles = torch.arange(-proportion, proportion, 10)  # [-90, -80, ..., 80]
        self.discrete_angles = self.discrete_angles / 180.0 * math.pi  # 转换为弧度

        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)
        alphas = alphas * self.proportion_alpha

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        
        angles = angles * self.proportion
        angles = self.map_to_discrete_angles(angles)

        return alphas, angles

    def map_to_discrete_angles(self, angles):
        """
    将连续角度映射到最近的离散角度值。
    :param angles: 连续角度值，范围为 (-π/2, π/2)
    :return: 离散角度值，范围为 (-90°, 90°)，每隔 10°
    """
    # 将 angles 从弧度转换为度数
        angles_deg = angles * 180.0 / math.pi

    # 将 discrete_angles 移动到与 angles_deg 相同的设备
        discrete_angles_deg = self.discrete_angles.to(angles_deg.device) * 180.0 / math.pi

    # 找到每个角度最近的离散值
        nearest_indices = torch.argmin(torch.abs(angles_deg.unsqueeze(-1) - discrete_angles_deg), dim=-1)

    # 映射到离散角度
        discrete_angles = self.discrete_angles.to(angles_deg.device)[nearest_indices]

        return discrete_angles
    
    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)

