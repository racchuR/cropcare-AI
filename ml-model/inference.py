"""
CropCare AI - Model Inference Script
Standalone inference script for crop disease detection
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import json
import os
import argparse
from pathlib import Path

class CropDiseaseCNN(nn.Module):
    """CNN model for crop disease classification"""
    
    def __init__(self, num_classes=8, pretrained=True):
        super(CropDiseaseCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CropDiseasePredictor:
    """Crop disease prediction class"""
    
    def __init__(self, model_path, class_names_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class names
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Default class names
            self.class_names = [
                "Healthy",
                "Bacterial_Blight",
                "Brown_Spot",
                "Leaf_Blight",
                "Leaf_Scald",
                "Leaf_Spot",
                "Rust",
                "Smut"
            ]
        
        # Load model
        self.model = CropDiseaseCNN(num_classes=len(self.class_names), pretrained=False)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from: {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict(self, image_path, top_k=3):
        """Predict crop disease from image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                predictions = []
                for i in range(top_k):
                    predictions.append({
                        'disease': self.class_names[top_indices[i].item()],
                        'confidence': top_probs[i].item()
                    })
                
                return predictions
                
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")
    
    def predict_single(self, image_path):
        """Predict single most likely disease"""
        predictions = self.predict(image_path, top_k=1)
        return predictions[0]

def main():
    parser = argparse.ArgumentParser(description='Crop disease detection inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='saved_models/best_model.pth', help='Path to model file')
    parser.add_argument('--classes', type=str, default='saved_models/class_names.json', help='Path to class names file')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Initialize predictor
    predictor = CropDiseasePredictor(args.model, args.classes)
    
    # Make prediction
    try:
        predictions = predictor.predict(args.image, args.top_k)
        
        print(f"\nPrediction results for: {args.image}")
        print("=" * 50)
        
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['disease']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
        
        # Get treatment advice for top prediction
        top_prediction = predictions[0]
        print(f"\nTop Prediction: {top_prediction['disease']}")
        print(f"Confidence: {top_prediction['confidence']*100:.2f}%")
        
        # Simple treatment advice (in a real app, this would come from a knowledge base)
        if top_prediction['disease'] == 'Healthy':
            print("\nTreatment Advice: Your plant appears healthy! Continue with regular care.")
        else:
            print(f"\nTreatment Advice: Consult with an agricultural expert for proper treatment of {top_prediction['disease']}.")
            print("General recommendations:")
            print("- Remove affected plant parts")
            print("- Apply appropriate fungicide/bactericide")
            print("- Improve air circulation")
            print("- Monitor plant health regularly")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
