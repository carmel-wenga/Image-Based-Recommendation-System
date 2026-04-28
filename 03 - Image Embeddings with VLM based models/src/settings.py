import os
## The base dir of the project
PROJECT_DIR = "/home/projects/IBRS/Image-Based-Recommendation-System"

## The "etc" dir contains all the files not tracked in GitHub
ETC_DIR = os.path.join(PROJECT_DIR, "etc")

## The .zip of the dataset is provided in the repository, but it's unzipped in "etc".
DATASET_ZIP_FILE = os.path.join(PROJECT_DIR, "dataset.zip")
DATASET_DIR = os.path.join(ETC_DIR, "dataset")
DATASET_METADATA_TXT_FILE = os.path.join(DATASET_DIR, "metadata.txt")
DATASET_METADATA_JSON_FILE = os.path.join(DATASET_DIR, "metadata.json")
DATASET_METADATA_DIR = os.path.join(DATASET_DIR, "metadata")

## SRC DIR
SRC_DIR = os.path.join(PROJECT_DIR, "src")

# Map the model parameter to the correct Elasticsearch field
MODEL_FIELD_MAPPING = {
    "resnet": "image_features",
    "siglip": "image_features_siglip",
    "gemini": "image_features_gemini"
}