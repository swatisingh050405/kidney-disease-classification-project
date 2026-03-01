from kidney_disease_classifier import logger
from kidney_disease_classifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from kidney_disease_classifier.pipeline.prepare_base_model_pipeline2 import PrepareBaseModelPipeline


stage_name = "Data Ingestion Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {stage_name} started <<<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()
        logger.info(f">>>>>>> stage {stage_name} completed <<<<<<<\n\nx==========x")
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