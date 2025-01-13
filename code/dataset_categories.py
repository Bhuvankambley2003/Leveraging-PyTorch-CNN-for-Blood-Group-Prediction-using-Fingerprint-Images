import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),            # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset
dataset = ImageFolder(root='data', transform=transform)

# Print the classes and their indices
print("Classes:", dataset.classes)
print("Class Indices:", dataset.class_to_idx)

# Optional: Print the total number of samples in the dataset
print(f'Total number of samples: {len(dataset)}')
