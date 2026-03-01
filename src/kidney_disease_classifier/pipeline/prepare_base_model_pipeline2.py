from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.components.prepare_base_model import PrepareBaseModel
from kidney_disease_classifier import logger


STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelPipeline:
     def __init__(self):
         pass

     def main(self):
        config_manager = ConfigurationManager()
        prepare_base_model_config = config_manager.get_prepare_base_model_config()

        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # Load base model
        prepare_base_model.get_base_model()

        # Update base model (freeze + change classifier)
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        prepare_base_model_pipeline = PrepareBaseModelPipeline()
        prepare_base_model_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e