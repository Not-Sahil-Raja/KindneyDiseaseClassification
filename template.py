import os
from pathlib import Path
import logging

# logging String
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - : %(message)s")

project_name = "Kidney_Disease_Classification"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configurations.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entiy/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
]

for filepath in list_of_files:
    path = Path(filepath)
    filedr, filename = os.path.split(path)

    if filedr != "":
        os.makedirs(filedr, exist_ok=True)
        logging.info(f"Creating Directory; {filedr} for the file {filename}")

    if not os.path.exists(path) or (os.path.getsize(path) == 0):
        with open(path, "w") as f:
            pass
            logging.info(f"Creating empty file; {filename}")

    else:
        logging.info(f"File already exists; {filename}")
