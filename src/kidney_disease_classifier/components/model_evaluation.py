import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
import dagshub
dagshub.init(
     repo_owner='swatisingh050405', 
     repo_name='kidney-disease-classification-project', 
     mlflow=True)
       


class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
       
    def evaluate(self):

        print("Using device:", self.device)

        #  Load trained model
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, self.config.params_classes)

        model.load_state_dict(
            torch.load(self.config.trained_model_path, map_location=self.device)
        )

        model = model.to(self.device)
        model.eval()

        # Data Transform
        transform = transforms.Compose([
            transforms.Resize(tuple(self.config.params_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_dataset = datasets.ImageFolder(
            root=self.config.test_data_path / "test",
            transform=transform
        )

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        #  Accuracy
        test_accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100 

        print(f"Test Accuracy: {test_accuracy:.2f}%")

        #  Classification Report
        report = classification_report(all_labels, all_preds, output_dict=True)
        print(classification_report(all_labels, all_preds))

        #  Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = self.config.root_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        #  MLflow Logging
        with mlflow.start_run(run_name="VGG16_Evaluation"):

           mlflow.log_param("model_type", "VGG16")
           mlflow.log_param("image_size", self.config.params_image_size)
           mlflow.log_param("num_classes", self.config.params_classes)
           

           mlflow.log_metric("test_accuracy", test_accuracy)

           for label, metrics in report.items():
                 if isinstance(metrics, dict):
                      for metric_name, value in metrics.items():
                         mlflow.log_metric(f"{label}_{metric_name}", value)

           mlflow.log_artifact(cm_path)
           mlflow.log_artifact(self.config.trained_model_path)

        print(" Evaluation complete and logged to DagsHub MLflow")