from Kidney_Disease_Classification.constants import *
from Kidney_Disease_Classification.utils.common import read_yaml, create_directory
from Kidney_Disease_Classification.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(str(config_filepath))
        self.params = read_yaml(str(params_filepath))

        create_directory([self.config.artifacts_root])

    # Preparing the data ingestion configuration
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_file=config.local_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    # Pepraing the base model configuration
    def get_pepare_base_model_config(self) -> PrepareBaseModelConfig:

        config = self.config.prepare_base_model

        create_directory([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config
