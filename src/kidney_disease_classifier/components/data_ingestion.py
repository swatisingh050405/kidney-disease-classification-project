import os
import zipfile
import gdown
from kidney_disease_classifier import logger
from kidney_disease_classifier.utils.common import get_size
from kidney_disease_classifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self) -> str:
        """
        Download the data file from the source URL.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} to {zip_download_dir}")


            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)
       
            logger.info(f"Downloaded data to {zip_download_dir}")
            
        except Exception as e:
            logger.exception(f"Error occurred while downloading data: {e}")
            raise e
        
    
    def extract_zip_file(self) -> None:
        """
        Extract the downloaded zip file to the specified directory.
        """
        unzip_Path = self.config.unzip_dir
        os.makedirs(unzip_Path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_Path)