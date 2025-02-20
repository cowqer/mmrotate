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

# class RountingFunction(nn.Module):

#     def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
#         super().__init__()
#         self.kernel_number = kernel_number
#         self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
#                              groups=in_channels, bias=False)
#         self.norm = LayerNormProxy(in_channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

#         self.act_func = nn.Softsign()
#         self.proportion = proportion / 180.0 * math.pi
        
#         # init weights
#         trunc_normal_(self.dwc.weight, std=.02)
#         trunc_normal_(self.fc_alpha.weight, std=.02)
#         trunc_normal_(self.fc_theta.weight, std=.02)

#     def forward(self, x):

#         x = self.dwc(x)
#         x = self.norm(x)
#         x = self.relu(x)

#         x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

#         alphas = self.dropout1(x)
#         alphas = self.fc_alpha(alphas)
#         alphas = torch.sigmoid(alphas)

#         angles = self.dropout2(x)
#         angles = self.fc_theta(angles)
#         angles = self.act_func(angles)
#         angles = angles * self.proportion

#         return alphas, angles

#     def extra_repr(self):
#         s = (f'kernel_number={self.kernel_number}')
#         return s.format(**self.__dict__)

class RountingFunction1(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.dwc5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=1,
                             groups=in_channels, bias=False)
        self.dwc7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=1,
                             groups=in_channels, bias=False)
        
        self.norm = LayerNormProxy(in_channels)
        
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_shared = nn.Conv2d(in_channels, kernel_number * 2, kernel_size=1, bias=True)


        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc3.weight, std=.02)
        trunc_normal_(self.dwc5.weight, std=.02)
        trunc_normal_(self.dwc7.weight, std=.02)
        trunc_normal_(self.fc_shared.weight, std=.02)

    def forward(self, x):

        x = self.dwc3(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas, angles = torch.split(x, self.kernel_number, dim=1)
        alphas = torch.sigmoid(alphas)
        angles = self.act_func(angles)
        
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)
    
class RountingFunction1(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        
        self.norm = LayerNormProxy(in_channels)
        
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_shared = nn.Conv2d(in_channels, kernel_number * 2, kernel_size=1, bias=True)


        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc3.weight, std=.02)
        trunc_normal_(self.fc_shared.weight, std=.02)

    def forward(self, x):

        x = self.dwc3(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas, angles = torch.split(x, self.kernel_number, dim=1)
        alphas = torch.sigmoid(alphas)
        angles = self.act_func(angles)
        
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )
        
class RountingFunction2(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.InceptionDWConv2d = InceptionDWConv2d(in_channels)
        
        self.dwc3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        
        self.norm = LayerNormProxy(in_channels)
        
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_shared = nn.Conv2d(in_channels, kernel_number * 2, kernel_size=1, bias=True)


        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc3.weight, std=.02)
        trunc_normal_(self.dwc5.weight, std=.02)
        trunc_normal_(self.dwc7.weight, std=.02)
        trunc_normal_(self.fc_shared.weight, std=.02)

    def forward(self, x):

        # x = self.dwc3(x)
        x = self.InceptionDWConv2d(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas, angles = torch.split(x, self.kernel_number, dim=1)
        alphas = torch.sigmoid(alphas)
        angles = self.act_func(angles)
        
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)