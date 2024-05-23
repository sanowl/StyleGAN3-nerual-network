import torch
import torch.optim as optim
from torch.nn import functional as F
from model import Generator, Discriminator
from dataset import get_dataloader
from utils import ExponentialMovingAverage

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