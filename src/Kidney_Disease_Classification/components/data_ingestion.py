import os
import zipfile
import gdown
from Kidney_Disease_Classification import logger
from Kidney_Disease_Classification.utils.common import get_size
from Kidney_Disease_Classification.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_data(self) -> str:
        """
        Download data from the source
        """

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(
                f"Downloading data from {dataset_url} into file {zip_download_dir}"
            )

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(
                f"Data downloaded at {dataset_url} into file {zip_download_dir}"
            )

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        Extract the zip file into the unzip directory
        Function returns None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
