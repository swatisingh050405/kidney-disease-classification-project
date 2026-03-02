from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path




@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    source_data_path: Path
    split_data_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int




@dataclass
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_augmentation: bool
    params_batch_size: int
    params_epochs: int
    params_image_size: list
    params_classes: int
    params_learning_rate: float