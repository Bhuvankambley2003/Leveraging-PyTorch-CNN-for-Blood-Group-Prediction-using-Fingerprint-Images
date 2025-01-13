import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class FingerprintCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(FingerprintCNN, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Third convolution block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize session state
if 'model' not in st.session_state:
    # Load the model
    st.session_state.model = FingerprintCNN(num_classes=8)
    try:
        checkpoint = torch.load('best_fingerprint_model.pth', map_location=torch.device('cpu'),weights_only=True)
        st.session_state.model.load_state_dict(checkpoint['model_state_dict'])
        st.session_state.model.eval()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """Preprocess the image for model input"""
    if image is not None:
        image = transform(image).unsqueeze(0)
    return image

def get_prediction(image):
    """Get prediction from model"""
    try:
        with torch.no_grad():
            outputs = st.session_state.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_prob, predicted_class = torch.max(probabilities, 1)
            return predicted_class.item(), predicted_prob.item()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def introduction():
    st.title("Blood Group Prediction Using PyTorch CNN")
    st.write("""
    Blood group detection plays a vital role in medical emergencies, transfusions, and organ donations. 
    Traditional methods require blood samples and laboratory testing, which can be time-consuming and invasive. 
    This project introduces a novel approach to non-invasive blood group detection using fingerprint images.
    
    ### How it works:
    1. Upload a fingerprint image
    2. The deep learning model analyzes the unique patterns
    3. Get an instant prediction of the blood group
    
    ### Technology Used:
    - Deep Convolutional Neural Network (CNN)
    - PyTorch Framework
    - Advanced Image Processing
    
    ### Note:
    This is a research project and should not be used as a substitute for professional medical testing.
    Always consult healthcare professionals for medical decisions.
    """)

def prediction():
    st.title("Blood Group Prediction")
    st.write("Upload a fingerprint image to predict the blood group")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        col1.image(image, caption='Uploaded Fingerprint', use_column_width=True)
        
        # Add a prediction button
        if col1.button('Predict Blood Group'):
            with st.spinner('Analyzing fingerprint...'):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Get prediction
                predicted_class, confidence = get_prediction(processed_image)
                
                if predicted_class is not None:
                    # Map the predicted class to blood group
                    blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
                    predicted_blood_group = blood_groups[predicted_class]
                    
                    # Display results
                    col2.markdown("### Results:")
                    col2.markdown(f"**Predicted Blood Group:** {predicted_blood_group}")
                    col2.markdown(f"**Confidence Score:** {confidence*100:.2f}%")
                    
                    # Add confidence visualization
                    col2.progress(confidence)
                    
                    if confidence < 0.7:
                        col2.warning("⚠️ Low confidence prediction. Consider uploading a clearer image.")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Prediction"])

# Display the selected page
if page == "Introduction":
    introduction()
else:
    prediction()

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application is for research purposes only. "
    "The predictions should not be used for medical decisions."
)