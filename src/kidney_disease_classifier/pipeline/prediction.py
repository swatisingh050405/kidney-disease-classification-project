import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
from pathlib import Path

from kidney_disease_classifier.utils.common import decodeImage
from kidney_disease_classifier.utils.common import load_bin


class PredictionPipeline:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #  Model path
        self.model_path = Path("artifacts/training/trained_model.pth")

        #  Load model
        self.model = models.vgg16()
        self.model.classifier[6] = nn.Linear(4096, 4)

        # Using your load_bin utility
        self.model.load_state_dict(
            load_bin(self.model_path)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        #  Image Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.class_labels = ["Cyst", "Normal", "Stone", "Tumor"]

    def predict(self, image_base64: str):

        #  Decode base64 image and save temporarily
        temp_image_path = "temp_image.png"
        decodeImage(image_base64, temp_image_path)

        #  Load image
        image = Image.open(temp_image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        #  Predict
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_labels[predicted.item()]
        confidence_score = round(confidence.item() * 100, 2)

        return {
            "prediction": predicted_class,
            "confidence": confidence_score
        }