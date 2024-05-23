import torch
from model import Generator, Discriminator
from dataset import get_dataloader
from train import train

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set hyperparameters
    latent_dim = 512
    hidden_dim = 512
    output_channels = 3
    num_layers = 8
    num_epochs = 100
    batch_size = 32
    num_workers = 4
    
    # Initialize generator and discriminator
    generator = Generator(latent_dim, hidden_dim, output_channels, num_layers).to(device)
    discriminator = Discriminator(output_channels, hidden_dim, num_layers).to(device)
    
    # Get dataloader
    dataset_path = 'path_to_your_dataset'
    dataloader = get_dataloader(dataset_path, batch_size, num_workers)
    
    # Train the GAN
    train(generator, discriminator, dataloader, num_epochs, latent_dim, device)
    
    # Save the trained generator
    torch.save(generator.state_dict(), 'generator.pth')
    
    print("Training completed. Generator saved as 'generator.pth'.")

if __name__ == '__main__':
    main()