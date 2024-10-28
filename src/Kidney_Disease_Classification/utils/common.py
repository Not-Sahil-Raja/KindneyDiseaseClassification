import os
from box.exceptions import BoxValueError
import yaml
from Kidney_Disease_Classification import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(file_path: str) -> ConfigBox:
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {file_path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise BoxValueError(f"Error in reading yaml file: {file_path}")
    except Exception as e:
        raise e


@ensure_annotations
def create_directory(path_to_directories: list, verbose=True):
    for directory in path_to_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if verbose:
                logger.info(f"Directory created for path: {directory}")
        else:
            if verbose:
                logger.info(f"Directory already exists for path: {directory}")


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)

    logger.info(f"Json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"Binary file loaded succesfully from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imagestr, filename):
    image = base64.b64decode(image)
    with open(filename, "wb") as f:
        f.write(image)
        f.close()


def encodeImageIntoBase64(croppedImagePaths):
    with open(croppedImagePaths, "rb") as img_file:
        return base64.b64encode(img_file.read())
