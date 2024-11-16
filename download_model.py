import gdown
import zipfile


# downloading the model from google drive
def download_model_from_gdrive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)


DRIVE_URL = (
    "https://drive.google.com/file/d/1aSBsINRxalELJPE0KJMlZ9Qh6uXqBfqy/view?usp=sharing"
)
OUTPUT_PATH = "model/"
download_model_from_gdrive(DRIVE_URL, OUTPUT_PATH)

# extract the zip file

with zipfile.ZipFile("model/model_v1.zip", "r") as zip_ref:
    zip_ref.extractall("model/")

print("Model downloaded and extracted successfully")
