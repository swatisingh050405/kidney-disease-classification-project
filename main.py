from kidney_disease_classifier import logger
from kidney_disease_classifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from kidney_disease_classifier.pipeline.data_transformation_pipeline import DataTransformationPipeline
from kidney_disease_classifier.pipeline.prepare_base_model_pipeline2 import PrepareBaseModelPipeline
from kidney_disease_classifier.pipeline.model_training_pipeline import ModelTrainingPipeline
from kidney_disease_classifier.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline


if __name__ == "__main__":
    
    # ============================
    # Data Ingestion Stage
    # ============================

    try:
            stage_name = "Data Ingestion Stage"
            logger.info(f">>>>>>> stage {stage_name} started <<<<<<<")
            data_ingestion_pipeline = DataIngestionPipeline()
            data_ingestion_pipeline.main()
            logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e

    
    # ============================
    # Data Transformation Stage
    # ============================
    
    try:
        stage_name = "Data Transformation Stage"
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")

        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e



    # ============================
    # Prepare Base Model Stage
    # ============================


    try:
        stage_name = "Prepare Base Model Stage"
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")

        prepare_base_model_pipeline = PrepareBaseModelPipeline()
        prepare_base_model_pipeline.main()

        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

    
    # ============================
    # Model Training Stage
    # ============================

    
    try:
        stage_name = "Model Training Stage"
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")

        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()

        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e
    

    # ============================
    # Model Evaluation Stage
    # ============================
    
    try:

        stage_name = "Model Evaluation Stage"
        logger.info(f">>>>>>> stage {stage_name} started <<<<<<<<")

        obj = ModelEvaluationPipeline()
        obj.main()

        logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e