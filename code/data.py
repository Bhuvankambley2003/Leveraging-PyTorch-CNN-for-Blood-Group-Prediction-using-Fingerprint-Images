# Add this right after creating your dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.5], std=[0.5])  
])
# Print a few samples to verify
dataset = ImageFolder(root='data', transform=transform)
print("Classes found:", dataset.classes)
print("Class to index mapping:", dataset.class_to_idx)
for i in range(3):
    img, label = dataset[i]
    print(f"Sample {i}:")
    print(f"Label index: {label}")
    print(f"Class name: {dataset.classes[label]}")
    print(f"Image path: {dataset.samples[i][0]}\n")