import os
import random
import shutil
from pathlib import Path

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def split_data(self):

        source_dir = self.config.source_data_path
        dest_dir = self.config.split_data_path

        for split in ["train", "val", "test"]:
            for class_name in os.listdir(source_dir):
                class_path = source_dir / class_name
                if not class_path.is_dir():
                    continue
                os.makedirs(dest_dir / split / class_name, exist_ok=True)

        for class_name in os.listdir(source_dir):
            class_path = source_dir / class_name
            if not class_path.is_dir():
                continue

            images = os.listdir(class_path)
            random.shuffle(images)

            total = len(images)
            train_end = int(total * self.config.train_ratio)
            val_end = train_end + int(total * self.config.val_ratio)

            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]

            for img in train_images:
                shutil.copy(class_path / img, dest_dir / "train" / class_name / img)

            for img in val_images:
                shutil.copy(class_path / img, dest_dir / "val" / class_name / img)

            for img in test_images:
                shutil.copy(class_path / img, dest_dir / "test" / class_name / img)

        print("Data splitting completed successfully")