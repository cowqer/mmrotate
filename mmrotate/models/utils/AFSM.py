import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAlphaLayer(nn.Module):
    """Adaptive Alpha Layer to generate a weight map for feature fusion."""

    def __init__(self, in_channels):
        super(AdaptiveAlphaLayer, self).__init__()
        # Define a convolutional layer to produce a per-pixel weight map for each sample
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        """Generate adaptive alpha map for each input feature map."""
        # Calculate alpha for each spatial location and each batch sample
        alpha_map = self.conv(x)  # output shape: [batch_size, 1, height, width]
        alpha_map = torch.sigmoid(alpha_map)  # To keep alpha in the range [0, 1]
        return alpha_map
