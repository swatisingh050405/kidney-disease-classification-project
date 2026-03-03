import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import dagshub
import mlflow

dagshub.init(
    repo_owner="swatisingh050405",
    repo_name="kidney-disease-classification-project",
    mlflow=True
)

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(self):

        print("Using device:", self.device)

        with mlflow.start_run(run_name="VGG16_Training"):

            # ✅ Log Hyperparameters
            mlflow.log_param("model_type", "VGG16")
            mlflow.log_param("epochs", self.config.params_epochs)
            mlflow.log_param("batch_size", self.config.params_batch_size)
            mlflow.log_param("learning_rate", self.config.params_learning_rate)
            mlflow.log_param("image_size", self.config.params_image_size)
            mlflow.log_param("num_classes", self.config.params_classes)

            # ------------------------------------------------
            # Load Updated Base Model
            # ------------------------------------------------
            model = models.vgg16()
            model.classifier[6] = nn.Linear(
                4096,
                self.config.params_classes
            )

            model.load_state_dict(
                torch.load(
                    self.config.updated_base_model_path,
                    map_location=self.device
                )
            )

            model = model.to(self.device)

            # ------------------------------------------------
            # Data Transforms
            # ------------------------------------------------
            transform = transforms.Compose([
                transforms.Resize(tuple(self.config.params_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            train_dataset = datasets.ImageFolder(
                root=self.config.training_data / "train",
                transform=transform
            )

            val_dataset = datasets.ImageFolder(
                root=self.config.training_data / "val",
                transform=transform
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.params_batch_size,
                shuffle=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.params_batch_size,
                shuffle=False
            )

            # ------------------------------------------------
            # Loss & Optimizer
            # ------------------------------------------------
            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(
                model.classifier[6].parameters(),
                lr=self.config.params_learning_rate
            )

            epochs = self.config.params_epochs

            # ------------------------------------------------
            # Training Loop
            # ------------------------------------------------
            for epoch in range(epochs):

                model.train()
                train_loss = 0
                train_correct = 0

                for images, labels in train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()

                train_loss /= len(train_loader)
                train_accuracy = 100 * train_correct / len(train_dataset)

                model.eval()
                val_loss = 0
                val_correct = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = 100 * val_correct / len(val_dataset)

                print(
                    f"Epoch [{epoch+1}/{epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.2f}%"
                )

            # ✅ Log Metrics AFTER training completes
            mlflow.log_metric("final_train_accuracy", train_accuracy)
            mlflow.log_metric("final_val_accuracy", val_accuracy)

            # ✅ Save model
            torch.save(model.state_dict(), self.config.trained_model_path)

            # ✅ Log model artifact
            mlflow.log_artifact(self.config.trained_model_path)

            print("✅ Training Complete — Model Saved & Logged to MLflow")