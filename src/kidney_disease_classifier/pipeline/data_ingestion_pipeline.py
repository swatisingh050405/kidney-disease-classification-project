from kidney_disease_classifier.config import ConfigurationManager
from kidney_disease_classifier.components.data_ingestion import DataIngestion
from kidney_disease_classifier import logger

stage_name = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    
    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {stage_name} started <<<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()
        logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e