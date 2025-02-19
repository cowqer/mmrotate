import torch
import torch.nn as nn
import torch.nn.functional as F

### Adaptive feature selection module

class AdaptiveAlphaLayer(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveAlphaLayer, self).__init__()
        # 设计一个小的网络来生成 alpha
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 1)  # 输出一个标量 alpha
        
    def forward(self, x):
        # 通过一个小的网络来生成 alpha
        x = x.mean(dim=[2, 3])  # 对空间维度进行池化，得到每个通道的全局信息
        x = F.relu(self.fc1(x))
        alpha = torch.sigmoid(self.fc2(x))  # 输出在 [0, 1] 之间的值
        return alpha