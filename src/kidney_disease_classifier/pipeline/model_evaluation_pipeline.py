from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.components.model_evaluation import ModelEvaluation
from kidney_disease_classifier import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(config=eval_config)
        model_evaluation.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")

        obj = ModelEvaluationPipeline()
        obj.main()

        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e