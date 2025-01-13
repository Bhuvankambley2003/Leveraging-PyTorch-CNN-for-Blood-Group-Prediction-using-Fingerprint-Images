import torch
import torchvision.transforms as transforms
import os
from PIL import Image

# Load the trained model
class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 8)  # Adjust according to your number of classes

    def forward(self, x):
        x = self.pool(torch.nn.ReLU()(self.conv1(x)))
        x = self.pool(torch.nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model weights
model = CNNClassifier()
model.load_state_dict(torch.load('final_model_weights.pth', weights_only=True))
model.eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),            # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Directory containing the images for prediction
image_dir = r'testing data'  # Use raw string to avoid escape sequences

# Check if directory exists
if not os.path.exists(image_dir):
    print(f"Directory '{image_dir}' does not exist.")
    exit(1)

# Initialize predictions dictionary
predictions = {}

print(f"Looking for images in: {os.path.abspath(image_dir)}")
print("Files in directory:")
for filename in os.listdir(image_dir):
    print(f"Found file: {filename}")  # Print each file found
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Supported file types
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        img = transform(img).unsqueeze(0)  # Add batch dimension

        print(f"Processing image: {img_path}, Size after transform: {img.shape}")
        
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)  # Get the predicted class
            predictions[filename] = predicted.item()  # Store the class
            print(f'Predicted {predicted.item()} for {filename}')  # Print prediction for each image

# Check if predictions were made
if predictions:
    for img_file, pred_class in predictions.items():
        print(f'Image: {img_file}, Predicted Class: {pred_class}')
else:
    print("No predictions were made.")
