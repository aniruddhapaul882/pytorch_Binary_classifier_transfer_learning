import os
import torch
from torchvision import transforms
from PIL import Image
from data.dataloaders import DataLoaderSetup 
from config import Config as cfg
from torchvision import models
import numpy as np

def load_model(model_path, device):
    # Define the model (ResNet18 for binary classification)
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft.eval()
    return model_ft

def infer_image(image_path, model, device, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        pred = torch.where(outputs > cfg.PRED_THRESH, 1.0, 0.0)
    return pred.item()

def main(image_path):
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(cfg.WEIGHT_DIR, 'full_best.pt')
    model = load_model(model_path, device)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Preprocessing for ResNet
    ])
    
    
    prediction = infer_image(image_path, model, device, transform)
    
    print(f'Prediction for image {image_path}: {prediction}')

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'  # Replace with the actual image path
    main(image_path)
