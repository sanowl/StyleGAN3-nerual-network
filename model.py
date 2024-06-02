import torch
import torch.nn as nn
import torch.nn.functional as F

# Mapping Network
class MappingNetwork(nn.Module):
    """
    Mapping Network class for mapping latent vectors to intermediate latent space.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)])
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

# Noise Injection
class NoiseInjection(nn.Module):
    """
    Noise Injection class for adding random noise to feature maps.
    """
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.scale * noise    


# Adaptive Instance Normalization (AdaIN)
class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) class for applying style to feature maps.
    """
    def __init__(self, latent_dim, channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(latent_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)
        return (1 + gamma) * self.norm(x) + beta



# Self-Attention
class SelfAttention(nn.Module):
    """
    Self-Attention class for applying self-attention to feature maps.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

# Blur
class Blur(nn.Module):
    """
    Blur class for applying blurring to feature maps.
    """
    def __init__(self, channels):
        super(Blur, self).__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel[None, None, :, :] / kernel.sum()
        self.register_buffer('kernel', kernel.repeat(channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=x.shape[1])

# Style Layer
class StyleLayer(nn.Module):
    """
    Style Layer class for applying style and upsampling/downsampling to feature maps.
    """
    def __init__(self, latent_dim, in_channels, out_channels, kernel_size=3, upsample=False, attention=False):
        super(StyleLayer, self).__init__()
        self.noise_injection = NoiseInjection(out_channels)
        self.adain = AdaIN(latent_dim, out_channels)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
        self.act = nn.LeakyReLU(0.2)
        self.upsample = upsample
        if upsample:
            self.blur = Blur(out_channels)
        self.attention = SelfAttention(out_channels) if attention else None

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.blur(x)
        x = self.conv(x)
        x = self.noise_injection(x)
        x = self.adain(x, w)
        x = self.act(x)
        if self.attention:
            x = self.attention(x)
        return x

# Residual Block
class ResidualBlock(nn.Module):
    """
    Residual Block class for applying residual connections to feature maps.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.act2 = nn.LeakyReLU(0.2)
        self.downsample = downsample
        self.downsample_layer = nn.Conv2d(in_channels, out_channels, 1, stride=2) if downsample else None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
            residual = self.downsample_layer(residual)
        return self.act2(x + residual)

# Generator
class Generator(nn.Module):
    """
    Generator class for generating images from latent vectors.
    """
    def __init__(self, latent_dim, hidden_dim, output_channels, num_layers):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim, hidden_dim, num_layers)
        self.style_layers = nn.ModuleList([
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True, attention=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True, attention=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True),
            StyleLayer(hidden_dim, hidden_dim, hidden_dim, upsample=True),
            StyleLayer(hidden_dim, hidden_dim, output_channels, upsample=False)
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 1),
            nn.Tanh()
        )

    def forward(self, z):
        w = self.mapping(z)
        w = w.unsqueeze(1).repeat(1, len(self.style_layers), 1)
        x = torch.randn(z.shape[0], self.style_layers[0].conv.in_channels, 4, 4).to(z.device)
        for i, style_layer in enumerate(self.style_layers):
            x = style_layer(x, w[:, i])
        x = self.to_rgb(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    """
    Discriminator class for determining the realness of generated images.
    """
    def __init__(self, input_channels, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        ])

        for _ in range(num_layers - 1):
            self.layers.extend([
                ResidualBlock(hidden_dim, hidden_dim * 2, downsample=True),
                nn.LeakyReLU(0.2)
            ])
            hidden_dim *= 2

        self.layers.extend([
            nn.Conv2d(hidden_dim, 1, 4, stride=1, padding=0),
            nn.Flatten()
        ])

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
