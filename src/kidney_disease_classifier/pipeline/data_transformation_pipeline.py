from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.components.data_transformation import DataTransformation
from kidney_disease_classifier import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.split_data()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = DataTransformationPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e