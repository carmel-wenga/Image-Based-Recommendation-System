import os

## The base dir of the project
ENV_PROJECT_DIR = os.getcwd()

## The "etc" dir contains all the files not tracked in GitHub
ENV_ETC_DIR = os.path.join(ENV_PROJECT_DIR, "etc")

## The .zip of the dataset is provided in the repository, but it's unzipped in "etc"
ENV_DATASET_ZIP_FILE = os.path.join(ENV_PROJECT_DIR, "dataset.zip")
ENV_DATASET_DIR = os.path.join(ENV_ETC_DIR, "dataset")
ENV_DATASET_METADATA_JSON_FILE = os.path.join(ENV_DATASET_DIR, "metadata.json")
ENV_DATASET_METADATA_DIR = os.path.join(ENV_DATASET_DIR, "metadata")

# GoogleNet
ENV_GOOGLENET_MODEL_URL = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

## Folder where to save the tar file to download
ENV_GOOGLENET_DIR = os.path.join(ENV_ETC_DIR, "googlenet")

## Target file name of the downloaded GoogleNet Model
ENV_GOOGLENET_ARCHIVE_MODEL_FILE = os.path.join(ENV_GOOGLENET_DIR, "googlenet.tgz")

## GoogleNet model file
ENV_GOOGLENET_MODEL_GRAPH = os.path.join(ENV_GOOGLENET_DIR, "classify_image_graph_def.pb")

# Image Embeddings
ENV_IMAGE_FEATURES_DIR = os.path.join(ENV_ETC_DIR, "images_embeddings")

# nmslib
ENV_NMSLIB_DIR = os.path.join(ENV_ETC_DIR, "nmslib")
ENV_NMSLIB_INDEX_NAME = os.path.join(ENV_NMSLIB_DIR, "index.bin")