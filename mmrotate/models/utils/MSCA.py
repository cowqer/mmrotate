######################  MSCAAttention ####     START   by  AI&CV  ###############################
 
 
import torch
import torch.nn as nn
from torch.nn import functional as F


class MSCAAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):
        u = x.clone()
        
        attn = self.conv0(x)
 
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
 
        attn = self.conv3(attn)
 
        return attn * u

class MSCAAttention1(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):
        u = x.clone()
        
        attn = self.conv0(x)
        # print('attn:',attn.shape)
        
        attn_0_w = self.conv0_1(attn)
        attn_1_w = self.conv1_1(attn)
        attn_2_w = self.conv2_1(attn)
        attn_w = attn_0_w + attn_1_w + attn_2_w
        # print('attn_w:',attn_w.shape)
        
        attn_0_h = self.conv0_2(attn)
        attn_1_h = self.conv1_2(attn)
        attn_2_h = self.conv2_2(attn)
        attn_h = attn_0_h + attn_1_h + attn_2_h
        # print('attn_h:',attn_h.shape)
        
        alpha = 0.5
        attn = attn + alpha * attn_w + (1 - alpha) * attn_h
 
        attn = self.conv3(attn)
 
        return attn * u

class HWMSCAAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=0, groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=0, groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=0, groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=0, groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 15), padding=0, groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (15, 1), padding=0, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = F.pad(attn, (3, 3, 0, 0))  
        attn_0 = self.conv0_1(attn_0)

        attn_0 = F.pad(attn_0, (0, 0, 3, 3))  
        attn_0 = self.conv0_2(attn_0)

        attn_1 = F.pad(attn, (5, 5, 0, 0))  # 在上下侧填充5
        attn_1 = self.conv1_1(attn_1)

        attn_1 = F.pad(attn_1, (0, 0, 5, 5))  # 在左右侧填充5
        attn_1 = self.conv1_2(attn_1)

        attn_2 = F.pad(attn, (7, 7, 0, 0))  
        attn_2 = self.conv2_1(attn_2)

        attn_2 = F.pad(attn_2, (0, 0, 7, 7)) 
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
 
        attn = self.conv3(attn)
 
        return attn * u
 
class MSCAAttention2(nn.Module):

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        u = x.clone()
        
        attn = self.conv0(x)
 
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
 
        attn = self.conv3(attn)
        
        se = self.global_avg_pool(attn).view(attn.shape[0], -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(attn.shape[0], attn.shape[1], 1, 1)
 
        return attn * u * se

class MSCAAttention3(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):
        u = x.clone()
        
        attn = self.conv0(x)

        attn_0_w = self.conv0_1(attn)
        attn_1_w = self.conv1_1(attn)
        attn_2_w = self.conv2_1(attn)
        attn_w = attn_0_w + attn_1_w + attn_2_w
        attn_w = self.conv3(attn_w)
        
        attn_0_h = self.conv0_2(attn)
        attn_1_h = self.conv1_2(attn)
        attn_2_h = self.conv2_2(attn)
        attn_h = attn_0_h + attn_1_h + attn_2_h
        attn_h = self.conv3(attn_h)

        # alpha = 0.5
        # attn = attn + alpha * attn_w + (1 - alpha) * attn_h
        attn = attn + attn_w + attn_h
 
        attn = self.conv3(attn)
 
        return attn * u
    
class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x) 

    
class MSCAAttention4(MSCAAttention):
    def __init__(self, dim):
        super().__init__(dim)
        
        self.SA = SpatialAttention()
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 9), groups=dim, dilation=3)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(9, 0), groups=dim, dilation=3)

 
    def forward(self, x):
        u = x.clone()
        
        attn = self.conv0(x)
 
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.SA(attn) * attn

        attn = self.conv3(attn)
        
        return attn * u
    
class MSCAAttention5(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x):
        u = x.clone()
        
        attn = self.conv0(x)
 
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        
        attn = attn + attn_0 + attn_1 + attn_2
 
        attn = self.conv3(attn)
 
        return attn * u

class MSCAAttention6(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

        # **可学习的权重参数**
        self.scale_weights = nn.Parameter(torch.ones(3))  

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        # **计算不同尺度的注意力**
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        # **归一化可学习权重**
        scale_weights = F.softmax(self.scale_weights, dim=0)

        # **加权融合不同尺度**
        attn = scale_weights[0] * attn_0 + scale_weights[1] * attn_1 + scale_weights[2] * attn_2

        attn = self.conv3(attn)

        return attn * u

if __name__ == '__main__':

    dim = 64
    model = MSCAAttention4(64)

    # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(1, 64, 32, 32)

    # 前向传播
    output = model(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)

    # 检查输出内容
    # print("Output:", output)
     