import torch
import torch.nn as nn
import torch.nn.functional as F

class RFRModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFRModule, self).__init__()
        
        # Define the 1D convolutions
        self.conv_1x5 = nn.Conv2d(in_channels, out_channels, (1, 5), padding=(0, 2))
        self.conv_5x1 = nn.Conv2d(in_channels, out_channels, (5, 1), padding=(2, 0))
        
        # Learnable weight parameters λ1 and λ2
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, Freg):
        """
        Forward pass for the RFR module.
        
        Args:
            Freg (Tensor): Input regression feature map of shape [C, H, W]
        
        Returns:
            Tensor: The refined regression feature map
        """
        # Check the shape of Freg
        print(f"Input Freg shape: {Freg.shape}")
        
        # Ensure that the input Freg has 3 dimensions: [C, H, W]
        if len(Freg.shape) == 3:
            # Downsampling Freg along height and width
            Fh_reg = Freg.mean(dim=1, keepdim=True)  # Downsample along height (dim=1)
            Fw_reg = Freg.mean(dim=2, keepdim=True)  # Downsample along width (dim=2)
        else:
            raise ValueError("Expected input tensor with 3 dimensions, but got different shape.")

        # Apply 1D convolutions (1x5 and 5x1)
        Fh_reg_star = self.conv_1x5(Fh_reg)  # Apply 1x5 convolution along height
        Fw_reg_star = self.conv_5x1(Fw_reg)  # Apply 5x1 convolution along width
        
        # Refine the features by adding them back to the original features
        Freg_refined = Freg + self.lambda1 * Fh_reg_star + self.lambda2 * Fw_reg_star
        
        return Freg_refined


# Example usage
if __name__ == "__main__":
    # Example feature map (C, H, W) where C = 64, H = 32, W = 32
    Freg = torch.randn(64, 32, 32)

    # Instantiate the RFR module
    rfr_module = RFRModule(in_channels=64, out_channels=64)

    # Forward pass
    Freg_refined = rfr_module(Freg)
    
    # Print the output shape to verify the result
    print(Freg_refined.shape)  # Expected shape: [64, 32, 32]
