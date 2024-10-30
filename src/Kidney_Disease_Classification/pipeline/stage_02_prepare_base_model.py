from Kidney_Disease_Classification.config.configurations import ConfigurationManager
from Kidney_Disease_Classification.components.prepare_base_model import PrepareBaseModel
from Kidney_Disease_Classification import logger

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_pepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()

        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = PrepareBaseModelPipeline()
        pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e
