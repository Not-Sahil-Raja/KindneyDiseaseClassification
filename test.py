import requests

# Set the URL
# url = "https://sahilraja-kidney-disease-classification.hf.space/predict/file"
url = "http://localhost:8080/predict/file"


# Set the file path
# file_path = "artifacts/data_ingestion/content/data/CT KIDNEY DATASET Normal, CYST, TUMOR and STONE/STONE/Stone- (24).jpg"
file_path = (
    "artifacts/data_ingestion/binary_classification/non_ct_scan_dataset/image_10.jpg"
)

# Send the POST request with the image file
with open(file_path, "rb") as image_file:
    response = requests.post(url, files={"file": image_file})

# Print the response
print(response.json())
