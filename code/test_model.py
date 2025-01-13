import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from model import CNNClassifier

# FILE: code/test_model.py

import torchvision.transforms as transforms

def test_model_accuracy():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),            # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])

    # Load dataset
    dataset = ImageFolder(root='data', transform=transform)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load('final_model_weights.pth'))
    model.eval()

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Testing step
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total * 100  # Convert to percentage
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {test_accuracy:.2f}%')

    # Assert accuracy is within an acceptable range (e.g., above 70%)
    assert test_accuracy > 70, f"Test accuracy is too low: {test_accuracy:.2f}%"

if __name__ == "__main__":
    test_model_accuracy()