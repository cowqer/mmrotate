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

class DMSRB(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(DMSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Conv2d(channels_in, channels_in, kernel_size_1, groups=channels_in,padding=1)
        self.conv_3_2 = nn.Conv2d(channels_in * 2, channels_in * 2, kernel_size_1, groups=channels_in * 2,padding=1)
        self.conv_5_1 = nn.Conv2d(channels_in, channels_in, kernel_size_2, groups=channels_in,padding=2)
        self.conv_5_2 = nn.Conv2d(channels_in * 2, channels_in * 2, kernel_size_2, groups=channels_in * 2,padding=2)
        self.confusion = nn.Conv2d(channels_in * 4, channels_out, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        trunc_normal_(self.conv_3_1.weight, std=.02)
        trunc_normal_(self.conv_3_2.weight, std=.02)
        trunc_normal_(self.conv_5_1.weight, std=.02)
        trunc_normal_(self.conv_5_2.weight, std=.02)
        

    def forward(self, x):
        input_1 = x

        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        
        output = self.confusion(input_3)
        output = output + x
        return output
    
class MRoutingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        # self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
        #                      groups=in_channels, bias=False)
        
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        self.Mdw = DMSRB(in_channels, in_channels)


        trunc_normal_(self.fc_theta.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        
    def forward(self, x):
        
        x = self.Mdw(x)
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

if __name__ == "__main__":
    # 创建一个 DMSRB 实例
    channels_in = 64
    channels_out = 64

    dmsrb = DMSRB(channels_in, channels_out)

    # 创建一个输入张量，形状为 (B, C, H, W)
    batch_size = 2
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels_in, height, width)

    # 前向传播
    output = dmsrb(input_tensor)

    # 打印输入和输出的形状
    print(f'Input shape: {input_tensor.shape}')
    print(f'Output shape: {output.shape}')