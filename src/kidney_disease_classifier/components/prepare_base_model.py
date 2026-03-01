import torch
from torchvision import models
from torch import nn
from pathlib import Path
from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        """
        Load pretrained VGG16
        """
        self.model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

        self.save_model(self.model, self.config.base_model_path)
        return self.model

    def update_base_model(self):
        """
        Freeze layers + modify classifier
        """

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final layer
        num_features = self.model.classifier[6].in_features

        self.model.classifier[6] = nn.Linear(
            in_features=num_features,
            out_features=self.config.params_classes
        )

        self.save_model(self.model, self.config.updated_base_model_path)

        return self.model

    @staticmethod
    def save_model(model: nn.Module, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)