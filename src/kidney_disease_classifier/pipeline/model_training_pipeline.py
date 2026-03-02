from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.entity.config_entity import TrainingConfig
from kidney_disease_classifier.components.model_training import ModelTrainer
from kidney_disease_classifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()

        model_trainer = ModelTrainer(config=training_config)

        model_trainer.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e
