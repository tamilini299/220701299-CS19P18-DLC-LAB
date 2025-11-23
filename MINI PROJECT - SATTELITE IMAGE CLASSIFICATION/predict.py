import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load model
def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 3 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction function
def predict_image(image_data, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    image = Image.open(image_data).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100

    class_names = ['AnnualCrop', 'Forest', 'Highway', 'Residential', 'SeaLake']
    return class_names[predicted_idx], confidence
