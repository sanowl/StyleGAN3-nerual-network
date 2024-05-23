import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_dataloader(dataset_path, batch_size, num_workers):
    dataset = ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader