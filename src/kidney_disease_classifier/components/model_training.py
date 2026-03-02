import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(self):

        print("Using device:", self.device)

        # ------------------------------------------------
        # Load Updated Base Model (from Prepare Stage)
        # ------------------------------------------------
        model = models.vgg16()

        # Replace final classifier layer
        model.classifier[6] = nn.Linear(
            4096,
            self.config.params_classes
        )

        # Load saved updated base model weights
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

        # ------------------------------------------------
        # Load Dataset
        # ------------------------------------------------
        full_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size]
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

            # ---------------- TRAIN ----------------
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

            # ---------------- VALIDATION ----------------
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

        # ------------------------------------------------
        # Save Final Trained Model
        # ------------------------------------------------
        torch.save(
            model.state_dict(),
            self.config.trained_model_path
        )

        print("✅ Training Complete — Model Saved Successfully")