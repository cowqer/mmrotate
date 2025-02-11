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
    

class RountingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
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
class RountingFunctionPro(nn.Module):
### 2.10：只添加了maxpool
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0, reduction=32):
        """
        Initialize the RountingFunctionPro module.

        Args:
            in_channels (int): Number of input channels.
            kernel_number (int): Number of kernels.
            dropout_rate (float, optional): Dropout rate. Default is 0.2.
            proportion (float, optional): Proportion parameter. Default is 40.0.
            reduction (int, optional): Reduction factor for channels. Default is 32.
        """
        super().__init__()
        self.kernel_number = kernel_number

        # Depthwise convolution with kernel size 3x3
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.sa = SpatialAttention()
        # Layer normalization
        self.norm = LayerNormProxy(in_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Adaptive average pooling to 1x1
        ############################################
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        ############################################
        
        # Dropout and fully connected layer for alpha
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels*2, kernel_number, bias=True)

        # Dropout and fully connected layer for theta
        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels*2, kernel_number, bias=False)

        # Activation function
        self.act_func = nn.Softsign()
        
        # Convert proportion to radians
        self.proportion = proportion / 180.0 * math.pi
        
        # Initialize weights with truncated normal distribution
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        avg_x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        max_x = self.max_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        
        x = torch.cat([avg_x, max_x], dim=-1)  # 形状变为 [batch_size, 2 * in_channels]
        
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
    
class RountingFunction1(nn.Module):
### 2.10：添加了maxpool 和 STN
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0, reduction=32):
        """
        Initialize the RountingFunctionPro module.

        Args:
            in_channels (int): Number of input channels.
            kernel_number (int): Number of kernels.
            dropout_rate (float, optional): Dropout rate. Default is 0.2.
            proportion (float, optional): Proportion parameter. Default is 40.0.
            reduction (int, optional): Reduction factor for channels. Default is 32.
        """
        super().__init__()
        self.kernel_number = kernel_number

        # Depthwise convolution with kernel size 3x3
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.sa = SpatialAttention()
        # Layer normalization
        self.norm = LayerNormProxy(in_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Adaptive average pooling to 1x1
        ############################################
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        ############################################
        
        # Dropout and fully connected layer for alpha
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels*2, kernel_number, bias=True)

        # Dropout and fully connected layer for theta
        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels*2, kernel_number, bias=False)

        # Activation function
        self.act_func = nn.Softsign()
        
        # Convert proportion to radians
        self.proportion = proportion / 180.0 * math.pi
        
        # Initialize weights with truncated normal distribution
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        avg_x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        max_x = self.max_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        
        x = torch.cat([avg_x, max_x], dim=-1)  # 形状变为 [batch_size, 2 * in_channels]
        
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