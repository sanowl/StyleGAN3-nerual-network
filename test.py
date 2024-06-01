
import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size):
        super(FourierFeatures, self).__init__()
        # Random matrix B with dimensions (input_dim, mapping_size)
        self.B = torch.randn((input_dim, mapping_size)) * 10

    def forward(self, x):
        # Project the input coordinates using matrix B
        x_proj = 2 * torch.pi * x @ self.B
        # Apply sine and cosine functions
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Example usage
coords = torch.randn((16, 2))  # 16 points in 2D space
fourier_features = FourierFeatures(2, 64)  # Map to 64 dimensions
encoded_coords = fourier_features(coords)
print(encoded_coords.shape)  # Output shape will be (16, 128)
 