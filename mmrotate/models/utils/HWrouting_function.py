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
    
class HWRoutingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0, lambda_1 = 0.05, lambda_2 = 0.05):
        super().__init__()
        self.kernel_number = kernel_number
        
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None)) 
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1)) 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0))
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        
        branch1 = self.avg_pool_w(x)  # (batch_size, C, 1, W)
        branch1 = self.conv1(branch1)  # (batch_size, C, 1, W)
        branch1 = branch1.expand(-1, -1, x.size(2), -1)  # 扩展回 (batch_size, C, H, W)

        branch2 = self.avg_pool_h(x)  # (batch_size, C, H, 1)
        branch2 = self.conv2(branch2)  # (batch_size, C, H, 1)
        branch2 = branch2.expand(-1, -1, -1, x.size(3))  # 扩展回 (batch_size, C, H, W)
        
        x = self.lambda_1 * branch1 + self.lambda_2 * branch2 + x

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

