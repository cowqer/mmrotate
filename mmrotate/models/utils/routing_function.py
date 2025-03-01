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
        attention  = torch.cat([avg_out, max_out], dim=1)
        attention  = self.conv1(attention)
        return x * self.sigmoid(attention)

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

        self.dropout2 = nn.Dropout(dropout_rate)
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

class RountingFunction_AMpool(RountingFunction):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_alpha = nn.Linear(in_channels * 2, kernel_number, bias=True)
        self.fc_theta = nn.Linear(in_channels * 2, kernel_number, bias=False)

    def forward(self, x):
        # Apply spatial attention mechanism

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

class RountingFunction_stn(RountingFunction):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Apply spatial attention mechanism
        
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.spatial_attention(x)
        
        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles
    
class RountingFunction_stn_group(RountingFunction):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.sa = SpatialAttention(7)

        # 3 个卷积层用于计算 alphas
        self.fc_alpha1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.fc_alpha2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.fc_alpha3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 3 个卷积层用于计算 angles
        self.fc_theta1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.fc_theta2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.fc_theta3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Apply spatial attention mechanism
        attention = self.sa(x)
        x = x * attention

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x)  # x.shape = [batch_size, Cin, 1, 1]
        x = self.dropout1(x)

        # 计算 3 组 alphas
        alphas1 = self.fc_alpha1(x).squeeze(dim=-1).squeeze(dim=-1)
        alphas2 = self.fc_alpha2(x).squeeze(dim=-1).squeeze(dim=-1)
        alphas3 = self.fc_alpha3(x).squeeze(dim=-1).squeeze(dim=-1)

        # 计算 3 组 angles
        angles1 = self.fc_theta1(x).squeeze(dim=-1).squeeze(dim=-1)
        angles2 = self.fc_theta2(x).squeeze(dim=-1).squeeze(dim=-1)
        angles3 = self.fc_theta3(x).squeeze(dim=-1).squeeze(dim=-1)

        # 应用 sigmoid 激活到 alphas
        alphas1 = torch.sigmoid(alphas1)
        alphas2 = torch.sigmoid(alphas2)
        alphas3 = torch.sigmoid(alphas3)

        # 应用自定义激活函数到 angles
        angles1 = self.act_func(angles1) * self.proportion
        angles2 = self.act_func(angles2) * self.proportion
        angles3 = self.act_func(angles3) * self.proportion

        return (alphas1, alphas2, alphas3), (angles1, angles2, angles3)

    
class RountingFunction_stn_group_chunk(RountingFunction):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__(in_channels, kernel_number, dropout_rate, proportion)
        self.sa = SpatialAttention(7)

        self.fc = nn.Conv2d(in_channels, kernel_number, kernel_size=1, groups=kernel_number, bias=False)
        self.fc_combined = nn.Linear(in_channels, 2 * kernel_number)

    def forward(self, x):
    # Apply spatial attention mechanism
        attention = self.sa(x)
        x = x * attention

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        # Combine the two fully connected layers into one
        combined = self.dropout1(x)
        combined = self.fc_combined(combined)  # [batch_size, 2 * kernel_number] (combining alpha and theta)

        # Split the combined output into alphas and angles
        alphas, angles = torch.chunk(combined, 2, dim=-1)  # Split along the last dimension

        alphas = torch.sigmoid(alphas)  # [batch_size, kernel_number]
    
        angles = self.act_func(angles)  # [batch_size, kernel_number]
        angles = angles * self.proportion  # Adjust based on proportion

        return alphas, angles
    
    

class RountingFunction_promax(nn.Module):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.1, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.activation = nn.SiLU()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # SE attention
        self.se_fc1 = nn.Linear(in_channels, in_channels // 4, bias=False)
        self.se_fc2 = nn.Linear(in_channels // 4, in_channels, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        # Multi-layer MLP for better expressiveness
        self.fc_alpha_hidden = nn.Linear(in_channels, in_channels // 2)
        self.fc_theta_hidden = nn.Linear(in_channels, in_channels // 2)
        self.fc_alpha = nn.Linear(in_channels // 2, kernel_number, bias=True)
        self.fc_theta = nn.Linear(in_channels // 2, kernel_number, bias=False)

        self.act_func = nn.Tanh()  # Replacing softsign with tanh
        self.proportion = proportion / 180.0 * math.pi
        
        # Improved initialization
        nn.init.kaiming_normal_(self.dwc.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_alpha.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_theta.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.dwc(x)
        x = self.norm(x)
        x = self.activation(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)

        # SE attention
        att = torch.relu(self.se_fc1(x))
        att = torch.sigmoid(self.se_fc2(att))
        x = x * att

        x = self.dropout(x)

        alphas = torch.relu(self.fc_alpha_hidden(x))
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = torch.relu(self.fc_theta_hidden(x))
        angles = self.fc_theta(angles)
        angles = self.act_func(angles) * self.proportion

        return alphas, angles
    
class RoutingFunction_mlp(nn.Module):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha_hidden = nn.Linear(in_channels, in_channels // 2)  # 增加隐藏层
        self.fc_alpha = nn.Linear(in_channels // 2, kernel_number, bias=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_theta_hidden = nn.Linear(in_channels, in_channels // 2)  # 增加隐藏层
        self.fc_theta = nn.Linear(in_channels // 2, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi

        # 初始化权重
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # 形状: [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = torch.relu(self.fc_alpha_hidden(alphas))  # 先通过隐藏层
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = torch.relu(self.fc_theta_hidden(angles))  # 先通过隐藏层
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles
    
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SingleHeadSelfAttention, self).__init__()
        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape = [batch_size, channels, height, width]
        batch_size, C, H, W = x.size()

        # Flatten the spatial dimensions
        x_flat = x.view(batch_size, C, H * W).permute(0, 2, 1) # [batch_size, C, H*W]
        
        # Query, Key, Value projections
        Q = self.query(x_flat)  # [batch_size, C, H*W]
        K = self.key(x_flat)  # [batch_size, C, H*W]
        V = self.value(x_flat)  # [batch_size, C, H*W]

        # Calculate attention scores
        attention_scores = torch.bmm(Q.permute(0, 2, 1), K)  # [batch_size, H*W, H*W]
        attention_scores = self.softmax(attention_scores / (C ** 0.5))  # Scaled dot-product attention

        # Apply attention to values
        output = torch.bmm(attention_scores, V.permute(0, 2, 1))  # [batch_size, H*W, C]
        
        # Reshape back to original dimensions
        # output = output.permute(0, 2, 1).view(batch_size, C, H, W)  # [batch_size, C, H, W]
        output = output.permute(0, 2, 1).reshape(batch_size, C, H, W)
        return output
    
class RoutingFunction_Transformer(nn.Module):
    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Replace spatial attention with single-head self-attention
        self.self_attention = SingleHeadSelfAttention(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi

        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        
        x = self.self_attention(x)# Apply single-head self-attention
        
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles
    