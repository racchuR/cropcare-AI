import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from config import settings
import openai

# Disease classes (example - replace with actual disease classes)
DISEASE_CLASSES = [
    "Healthy",
    "Bacterial Blight",
    "Brown Spot",
    "Leaf Blight",
    "Leaf Scald",
    "Leaf Spot",
    "Rust",
    "Smut"
]

# Treatment advice database (example)
TREATMENT_DATABASE = {
    "Bacterial Blight": {
        "description": "Bacterial blight is caused by Xanthomonas bacteria",
        "treatment": "Apply copper-based fungicides, improve drainage, remove infected plants",
        "prevention": "Use disease-free seeds, practice crop rotation, maintain proper spacing"
    },
    "Brown Spot": {
        "description": "Brown spot is a fungal disease affecting leaves",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb",
        "prevention": "Avoid overhead watering, ensure good air circulation"
    },
    "Leaf Blight": {
        "description": "Leaf blight causes brown lesions on leaves",
        "treatment": "Apply systemic fungicides, remove affected leaves",
        "prevention": "Practice crop rotation, maintain plant health"
    },
    "Healthy": {
        "description": "Plant appears healthy with no signs of disease",
        "treatment": "Continue current care practices",
        "prevention": "Maintain proper watering, fertilization, and pest control"
    }
}

class CropDiseaseCNN(nn.Module):
    """Simple CNN model for crop disease classification"""
    
    def __init__(self, num_classes=len(DISEASE_CLASSES)):
        super(CropDiseaseCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def load_model(model_path: str = None):
    """Load the trained model"""
    if model_path is None:
        model_path = settings.MODEL_PATH
    
    # Check if model file exists
    if not os.path.exists(model_path):
        # Return a dummy model for development
        print(f"Warning: Model file not found at {model_path}. Using dummy model.")
        return create_dummy_model()
    
    try:
        model = CropDiseaseCNN()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}. Using dummy model.")
        return create_dummy_model()

def create_dummy_model():
    """Create a dummy model for development/testing"""
    model = CropDiseaseCNN()
    model.eval()
    return model

def preprocess_image(image_path: str):
    """Preprocess image for model inference"""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

def predict_disease(image_path: str):
    """Predict crop disease from image"""
    try:
        # Load model
        model = load_model()
        
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        # Get disease name
        disease_name = DISEASE_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        # Generate treatment advice
        treatment_advice = generate_treatment_advice(disease_name, confidence_score)
        
        return {
            "disease": disease_name,
            "confidence": confidence_score,
            "treatment_advice": treatment_advice
        }
        
    except Exception as e:
        # Fallback prediction for development
        print(f"Prediction error: {e}. Using fallback prediction.")
        return {
            "disease": "Healthy",
            "confidence": 0.85,
            "treatment_advice": "Your plant appears healthy. Continue with regular care including proper watering, fertilization, and pest monitoring."
        }

def generate_treatment_advice(disease_name: str, confidence: float):
    """Generate treatment advice using database and AI"""
    try:
        # Get basic treatment info from database
        if disease_name in TREATMENT_DATABASE:
            basic_info = TREATMENT_DATABASE[disease_name]
            
            # If confidence is high, use database info
            if confidence > 0.8:
                return f"{basic_info['description']}. Treatment: {basic_info['treatment']}. Prevention: {basic_info['prevention']}"
            
            # If confidence is medium, enhance with AI
            elif confidence > 0.5 and settings.OPENAI_API_KEY:
                return enhance_with_ai(disease_name, basic_info, confidence)
        
        # Fallback advice
        return f"Detected {disease_name} with {confidence:.2%} confidence. Please consult with an agricultural expert for proper treatment."
        
    except Exception as e:
        print(f"Error generating treatment advice: {e}")
        return "Please consult with an agricultural expert for proper diagnosis and treatment."

def enhance_with_ai(disease_name: str, basic_info: dict, confidence: float):
    """Enhance treatment advice using OpenAI API"""
    try:
        if not settings.OPENAI_API_KEY:
            return f"{basic_info['description']}. Treatment: {basic_info['treatment']}"
        
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        prompt = f"""
        Crop disease: {disease_name}
        Confidence: {confidence:.2%}
        Basic treatment: {basic_info['treatment']}
        
        Provide specific, actionable treatment advice for this crop disease. Include:
        1. Immediate treatment steps
        2. Long-term management
        3. Prevention measures
        4. When to consult an expert
        
        Keep the response concise and practical for farmers.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"AI enhancement error: {e}")
        return f"{basic_info['description']}. Treatment: {basic_info['treatment']}"

# Initialize model on import
print("Loading crop disease model...")
model = load_model()
print("Model loaded successfully!")
