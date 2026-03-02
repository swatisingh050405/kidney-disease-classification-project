from kidney_disease_classifier import logger
from kidney_disease_classifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from kidney_disease_classifier.pipeline.data_transformation_pipeline import DataTransformationPipeline
from kidney_disease_classifier.pipeline.prepare_base_model_pipeline2 import PrepareBaseModelPipeline
from kidney_disease_classifier.pipeline.model_training_pipeline import ModelTrainingPipeline




if __name__ == "__main__":
    stage_name = "Data Ingestion Stage"


    try:
            logger.info(f">>>>>>> stage {stage_name} started <<<<<<<")
            data_ingestion_pipeline = DataIngestionPipeline()
            data_ingestion_pipeline.main()
            logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e


    stage_name = "Data Transformation Stage"
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")

        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e




    stage_name = "Prepare Base Model Stage"
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")

        prepare_base_model_pipeline = PrepareBaseModelPipeline()
        prepare_base_model_pipeline.main()

        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e



    stage_name = "Model Training Stage"
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")

        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()

        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e