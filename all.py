import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from transformers import CLIPTokenizer, CLIPModel
import logging
from torch.cuda.amp import GradScaler, autocast
import json
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from a JSON file
def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file}' not found.")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from configuration file: {e}")
        raise

config = load_config('config.json')

# Neural network components
class MappingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.scale * noise

class AdaIN(nn.Module):
    def __init__(self, latent_dim, channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(latent_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)
        return (1 + gamma) * self.norm(x) + beta

class SelfAttention(nn.Module):
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

class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel[None, None, :, :] / kernel.sum()
        self.register_buffer('kernel', kernel.repeat(channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=x.shape[1])

class StyleLayer(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, kernel_size=3, upsample=False, attention=False):
        super(StyleLayer, self).__init__()
        self.noise_injection = NoiseInjection()
        self.adain = AdaIN(latent_dim, out_channels)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
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

class ResidualBlock(nn.Module):
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

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels, num_layers, clip_model):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim + 512, hidden_dim, num_layers)
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

    def forward(self, z, prompt_embedding):
        prompt_embedding = prompt_embedding.expand(z.size(0), -1)  # Ensure prompt_embedding matches batch size of z
        z = torch.cat((z, prompt_embedding), dim=1)
        w = self.mapping(z)
        w = w.unsqueeze(1).repeat(1, len(self.style_layers), 1)
        x = torch.randn(z.shape[0], self.style_layers[0].conv.in_channels, 4, 4).to(z.device)
        for i, style_layer in enumerate(self.style_layers):
            x = style_layer(x, w[:, i])
        x = self.to_rgb(x)
        return x

class Discriminator(nn.Module):
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

class ExponentialMovingAverage:
    def __init__(self, parameters, decay):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def update(self, parameters):
        for shadow_param, param in zip(self.shadow_params, parameters):
            shadow_param.data = (1.0 - self.decay) * param.data + self.decay * shadow_param.data

    def apply(self, parameters):
        for shadow_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(shadow_param.data)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_path_length_regularization(generator, z, fake_images):
    return 0.0

def compute_style_mixing_regularization(generator, z):
    return 0.0

def load_dataset(dataset_path, batch_size):
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path '{dataset_path}' does not exist.")
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    try:
        dataset = ImageFolder(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        logging.info(f"Loaded dataset with {len(dataset)} images.")
        return dataloader
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def extract_prompt_embedding(prompt, clip_model, clip_tokenizer, device, batch_size):
    try:
        with torch.no_grad():
            inputs = clip_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.expand(batch_size, -1)  # Expand to match batch size
        return text_features
    except Exception as e:
        logging.error(f"Error extracting prompt embedding: {e}")
        raise

def save_checkpoint(generator, discriminator, epoch):
    try:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")
        raise

def save_generated_images(generator, latent_dim, clip_model, clip_tokenizer, device, prompts, epoch):
    try:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]
            prompt_embedding = extract_prompt_embedding(prompt, clip_model, clip_tokenizer, device, z.size(0))
            fake_images = generator(z, prompt_embedding)
            fake_images = (fake_images + 1) / 2  # Denormalize
            grid = torchvision.utils.make_grid(fake_images, nrow=4)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title(f"Epoch {epoch+1} - Prompt: {prompt}")
            plt.savefig(f"generated_images_epoch_{epoch+1}.png")
            plt.close()
    except Exception as e:
        logging.error(f"Error saving generated images: {e}")
        raise

# Calculate FID
def calculate_fid(real_activations, fake_activations):
    try:
        mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
        mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    except Exception as e:
        logging.error(f"Error calculating FID: {e}")
        raise

# Extract activations from Inception model
def get_activations(model, dataloader, device):
    try:
        model.eval()
        activations = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                pred = model(images)[0]
                activations.append(pred.cpu().numpy())
        activations = np.concatenate(activations, axis=0)
        return activations
    except Exception as e:
        logging.error(f"Error extracting activations: {e}")
        raise

def train(generator, discriminator, dataloader, num_epochs, latent_dim, clip_model, clip_tokenizer, device, prompts):
    g_optim = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0.0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(0.0, 0.99))
    
    g_ema = ExponentialMovingAverage(generator.parameters(), decay=config['ema_decay'])
    scaler = GradScaler()  # For mixed precision training

    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def d_loss_fn(real_scores, fake_scores):
        return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
    
    def g_loss_fn(fake_scores):
        return -fake_scores.mean()

    best_fid = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        try:
            for i, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(device)
                batch_size = real_images.shape[0]
                
                # Train discriminator
                d_optim.zero_grad()
                
                with autocast():  # Mixed precision training
                    z = torch.randn(batch_size, latent_dim).to(device)
                    prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]
                    prompt_embedding = extract_prompt_embedding(prompt, clip_model, clip_tokenizer, device, batch_size)
                    
                    fake_images = generator(z, prompt_embedding)
                    real_scores = discriminator(real_images)
                    fake_scores = discriminator(fake_images.detach())
                    gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, device)
                    d_loss = d_loss_fn(real_scores, fake_scores) + config['gradient_penalty_weight'] * gradient_penalty

                scaler.scale(d_loss).backward()
                if (i + 1) % config['accumulate_grad_batches'] == 0:
                    scaler.step(d_optim)
                    scaler.update()
                
                # Train generator
                g_optim.zero_grad()
                
                with autocast():  # Mixed precision training
                    z = torch.randn(batch_size, latent_dim).to(device)
                    prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]
                    prompt_embedding = extract_prompt_embedding(prompt, clip_model, clip_tokenizer, device, batch_size)
                    
                    fake_images = generator(z, prompt_embedding)
                    fake_scores = discriminator(fake_images)
                    path_length_regularization = compute_path_length_regularization(generator, z, fake_images)
                    style_mixing_regularization = compute_style_mixing_regularization(generator, z)
                    g_loss = g_loss_fn(fake_scores) + config['path_length_weight'] * path_length_regularization + config['style_mixing_weight'] * style_mixing_regularization

                scaler.scale(g_loss).backward()
                if (i + 1) % config['accumulate_grad_batches'] == 0:
                    scaler.step(g_optim)
                    scaler.update()
                
                g_ema.update(generator.parameters())

            logging.info(f"Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
            
            # Checkpointing
            if (epoch + 1) % config['checkpoint_interval'] == 0:
                save_checkpoint(generator, discriminator, epoch)
                save_generated_images(generator, latent_dim, clip_model, clip_tokenizer, device, prompts, epoch)

            # Calculate FID
            real_activations = get_activations(inception, dataloader, device)
            fake_images = generator(torch.randn(len(real_activations), latent_dim).to(device), extract_prompt_embedding(prompts[0], clip_model, clip_tokenizer, device, len(real_activations)).to(device))
            fake_activations = get_activations(inception, fake_images, device)
            fid = calculate_fid(real_activations, fake_activations)
            logging.info(f"FID: {fid}")
            wandb.log({"FID": fid})

            # Early Stopping
            if fid < best_fid:
                best_fid = fid
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config['early_stopping_patience']:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    return

        except Exception as e:
            logging.error(f"Error during training epoch {epoch+1}: {e}")
            raise

if __name__ == '__main__':
    # Initialize models and move to device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        generator = Generator(config['latent_dim'], config['hidden_dim'], config['output_channels'], config['num_layers'], clip_model).to(device)
        discriminator = Discriminator(input_channels=3, hidden_dim=64, num_layers=4).to(device)
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        raise

    # Define prompts for conditioning
    prompts = [
        "a cat wearing sunglasses",
        "a cat wearing a hat",
        "a cat wearing a bowtie",
        "a cat wearing a scarf",
        "a cat wearing a shirt",
        "a cat wearing a sweater",
        "a cat wearing a jacket",
        "a cat wearing a dress",
        "a cat wearing a suit",
        "a cat wearing a costume"
    ]

    # Load dataset
    try:
        dataloader = load_dataset(config['dataset_path'], config['batch_size'])
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

    # Train the GAN
    try:
        train(generator, discriminator, dataloader, config['num_epochs'], config['latent_dim'], clip_model, clip_tokenizer, device, prompts)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise
