import torch
import torch.nn as nn
import torch.nn.functional as F

    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x

class GatedSPConv(nn.Module):
    ''' Pinwheel-shaped Convolution with Gating Mechanism '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.c2 = c2
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(p[i]) for i in range(4)])
        
        # Branch convolutions
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        
        # Gating mechanism (learns importance of each direction)
        # self.gate_fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),    # Global pooling to extract feature map statistics
        #     nn.Conv2d(c1, 4, kernel_size=1),  # Output 4 gate values
        #     nn.Sigmoid()  # Normalize between 0 and 1
        # )
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c2 // 4 * 4, kernel_size=1),  # 输出 [B, c2 // 4 * 4, 1, 1]
            nn.Sigmoid()
        )

        # Final fusion layer
        self.fusion = Conv(c2, c2, 2, s=1, p=0)
        self.ChannelShuffle = ChannelShuffle(groups=4)

    def forward(self, x):
        # Compute feature maps for each direction
        yw0 = self.cw(self.pad[0](x))  # Horizontal-1
        yw1 = self.cw(self.pad[1](x))  # Horizontal-2
        yh0 = self.ch(self.pad[2](x))  # Vertical-1
        yh1 = self.ch(self.pad[3](x))  # Vertical-2
        # print(yw0.shape, yw1.shape, yh0.shape, yh1.shape)
        # Compute gate weights
        gate = self.gate_fc(x)  # Shape: [B, 4, 1, 1]
        gate = gate.view(gate.shape[0], 4, self.c2 // 4, 1, 1)  # Reshape for broadcasting
        # print(gate.shape)
        # Apply learned gating weights to each branch
        yw0 = gate[:, 0] * yw0
        yw1 = gate[:, 1] * yw1
        yh0 = gate[:, 2] * yh0
        yh1 = gate[:, 3] * yh1
        # print(yw0.shape, yw1.shape, yh0.shape, yh1.shape)
        # Weighted sum instead of simple concatenation
        fused = torch.cat([yw0, yw1, yh0, yh1], dim=1)
        # print(fused.shape)
        fused = self.ChannelShuffle(fused)
        output = self.fusion(fused)
        
        # print(output.shape)
        return output

if __name__ == "__main__":
    # Create a random input tensor of shape (batch_size, channels, height, width)
    x = torch.randn(1, 3, 64, 64)  # 1 image, 3 channels, 64x64 size
    
    # Create an instance of PConv
    apconv = GatedPConv(c1=3, c2=128, k=3, s=1 )# output channels = 64
    
    # Forward pass
    output = apconv(x)
    
    # Print output shape
    print("Output shape:", output.shape)
