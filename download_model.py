import gdown
import zipfile


# downloading the model from google drive and unzipping it
def download_model_from_gdrive(drive_url, zip_download_dir, unzip_dir):
    model_drive_id = drive_url.split("/")[-2]
    prefix = "https://drive.google.com/uc?/export=download&id="
    gdown.download(prefix + model_drive_id, zip_download_dir)

    print(f"Model downloaded at {drive_url} into file {zip_download_dir}")

    with zipfile.ZipFile(zip_download_dir, "r") as zip_ref:
        zip_ref.extractall(unzip_dir)


DRIVE_URL = (
    "https://drive.google.com/file/d/1aSBsINRxalELJPE0KJMlZ9Qh6uXqBfqy/view?usp=sharing"
)
OUTPUT_PATH = "model/model.zip"
UNZIP_DIR = "model/"

download_model_from_gdrive(DRIVE_URL, OUTPUT_PATH, UNZIP_DIR)
