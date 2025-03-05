import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import trunc_normal_
from .MSCA import MSCAAttention, HWMSCAAttention, MSCAAttention1, MSCAAttention2, MSCAAttention3, MSCAAttention4, MSCAAttention5, MSCAAttention6

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
    
    ###adp0 为初始msca adp为hwmasca
class RountingFunction_MSCA(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.msca = MSCAAttention(in_channels)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)
 

    def forward(self, x):
        x = self.msca(x)
        
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)

class RountingFunction_attn(RountingFunction_MSCA):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.msca = MSCAAttention1(in_channels)
        
class RountingFunction_attn2(RountingFunction_MSCA):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.msca = MSCAAttention2(in_channels)

class RountingFunction_attn3(RountingFunction_MSCA):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.msca = MSCAAttention3(in_channels)

class RountingFunction_attn4(RountingFunction_MSCA):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.msca = MSCAAttention4(in_channels)

class RountingFunction_attn5(RountingFunction_MSCA):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.msca = MSCAAttention5(in_channels)
        
class RountingFunction_attn6(RountingFunction_MSCA):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.msca = MSCAAttention6(in_channels)