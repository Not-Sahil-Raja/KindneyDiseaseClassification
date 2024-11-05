from src.Kidney_Disease_Classification import logger
from Kidney_Disease_Classification.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from Kidney_Disease_Classification.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelPipeline,
)
from Kidney_Disease_Classification.pipeline.stage_03_model_training import (
    ModelTrainingPipeline,
)
from Kidney_Disease_Classification.pipeline.stage_04_model_evaluation import (
    EvaluationPipeline,
)


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    pipeline = PrepareBaseModelPipeline()
    pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f"************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evalution = EvaluationPipeline()
    model_evalution.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
