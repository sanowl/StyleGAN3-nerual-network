
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Mapping Network
class MappingNetwork(nn.Module):
    """
    Mapping Network class for mapping latent vectors to intermediate latent space.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mapping(x)

# Noise Injection
class NoiseInjection(nn.Module):
    """
    Noise Injection class for adding random noise to feature maps.
    """
    def __init__(self):
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
        self.noise_injection = NoiseInjection()
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
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.downsample = downsample
        self.downsample_layer = nn.Conv2d(in_channels, out_channels, 1, stride=2) if downsample else None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
            residual = self.downsample_layer(residual)
        return x + residual

# Generator
# Generator with Prompt Conditioning
class Generator(nn.Module):
    """
    Generator class for generating images from latent vectors and prompts.
    """
    def __init__(self, latent_dim, hidden_dim, output_channels, num_layers, clip_model):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim + clip_model.visual.output_dim, hidden_dim, num_layers)
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
        self.clip_model = clip_model
    
    def forward(self, z, prompt):
        # Encode the prompt using CLIP
        prompt_features = self.clip_model.encode_text(clip.tokenize(prompt).to(z.device))
        
        # Concatenate the latent vector and prompt features
        w = torch.cat([z, prompt_features], dim=1)
        
        w = self.mapping(w)
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

# Training loop with gradient penalty, path length regularization, and style mixing regularization
def train(generator, discriminator, dataloader, num_epochs, latent_dim, device):
    """
    Training loop for the GAN.
    """
    g_optim = optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))
    
    g_ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    
    # Use hinge loss for improved training stability
    def d_loss_fn(real_scores, fake_scores):
        return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
    
    def g_loss_fn(fake_scores):
        return -fake_scores.mean()
    
    for epoch in range(num_epochs):
        for real_images in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            
            # Train discriminator
            d_optim.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            
            real_scores = discriminator(real_images)
            fake_scores = discriminator(fake_images.detach())
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, device)
            
            d_loss = d_loss_fn(real_scores, fake_scores) + 10 * gradient_penalty
            d_loss.backward()
            d_optim.step()
            
            # Train generator
            g_optim.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            
            fake_scores = discriminator(fake_images)
            
            path_length_regularization = compute_path_length_regularization(generator, z, fake_images)
            style_mixing_regularization = compute_style_mixing_regularization(generator, z)
            
            g_loss = g_loss_fn(fake_scores) + 2 * path_length_regularization + 2 * style_mixing_regularization
            g_loss.backward()
            g_optim.step()
            
            g_ema.update(generator.parameters())
        
        # Evaluate and save checkpoints
        # ...

# Compute gradient penalty for WGAN-GP
def compute_gradient_penalty(discriminator, real_images, fake_images, device):
    """
    Computes the gradient penalty for WGAN-GP.
    """
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    interpolated_scores = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones(interpolated_scores.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

# Compute path length regularization
def compute_path_length_regularization(generator, z, fake_images):
    """
    Computes the path length regularization term for the generator.
    """
    path_lengths = torch.sqrt((fake_images ** 2).sum([2, 3])).mean(1)
    return ((path_lengths - path_lengths.mean()) ** 2).mean()

# Compute style mixing regularization
def compute_style_mixing_regularization(generator, z):
    """
    Computes the style mixing regularization term for the generator.
    """
    z2 = torch.randn_like(z)
    fake_images2 = generator(z2)
    mixed_fake_images = 0.5 * fake_images + 0.5 * fake_images2
    return ((mixed_fake_images - mixed_fake_images.mean()) ** 2).mean()

class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) class for model parameters.
    """
    def __init__(self, parameters, decay):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def update(self, parameters):
        for shadow_param, param in zip(self.shadow_params, parameters):
            shadow_param.data = self.decay * shadow_param.data + (1.0 - self.decay) * param.data

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = ImageFolder('path_to_your_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Initialize models and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 512
generator = Generator(latent_dim, hidden_dim=512, output_channels=3, num_layers=8).to(device)
discriminator = Discriminator(input_channels=3, hidden_dim=64, num_layers=4).to(device)

# Train the GAN
train(generator, discriminator, dataloader, num_epochs=100, latent_dim=latent_dim, device=device)

