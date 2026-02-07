# Build an Image-Based Recommendation System and Search Engine with ResNet, Elasticsearch, Python and Gemini

Medium Story: [Building an Image-Based Recommendation System and Search Engine with Deep Learning and Elasticsearch](https://medium.com/towards-artificial-intelligence/building-an-image-based-recommendation-system-and-search-engine-with-deep-learning-and-4bb96c4d9a64)

This second part of the repository focuses on building an Image-Based Recommendation System (IBRS) using the ResNet 
architecture for feature extraction and Elasticsearch as a vector database for storing and searching image embeddings. 

The three main focus areas of this part are:
1. Building an IBRS using ResNet and Elasticsearch,
2. Implementing image-based search capabilities,
3. Improving recommendation quality with image classification using the Gemini multimodal API

## Requirements
* Python 3.12
* Elasticsearch 9.1.8 via Docker
* Elasticsearch Python Client 9.2.0
* TensorFlow 2.16.1
* google-genai==1.61.0

## Project Setup Instructions
```shell
mkdir etc
python3 -m venv etc/venv-02
source etc/venv-02/bin/activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

## Jupyter Lab Setup
1. Configure jupyter to use the virtual environment that you have created as kernel
```sh
python -m ipykernel install --user --name IBRS-02 --display-name "Python3.12 (IBRS-02)"
```
2. Launch Jupyter Lab with the command below and choose ```Python3.12 (IBRS-02)``` as the kernel
```sh
jupyter lab
```

## Run Elasticsearch
Use the following command to run Elasticsearch in a Docker container:

```shell
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:9.1.8
```
The `xpack.security.enabled=false` property is used to disable security for local development. It allows connecting and interacting with 
Elasticsearch with http://localhost:9200 without authentication.

## The Notebook
The tutorial is implemented in the [IBRS and Search Engine.ipynb](./src/IBRS and Search Engine.ipynb) Jupyter Notebook.

## Related Medium Stories
* [Building an Image-Based Recommendation System and Search Engine with Deep Learning and Elasticsearch](https://medium.com/towards-artificial-intelligence/building-an-image-based-recommendation-system-and-search-engine-with-deep-learning-and-4bb96c4d9a64)

## Author
Carmel WENGA, Data & ML Engineer
- [Medium](https://medium.com/@carmelwenga)
- [LinkedIn](https://www.linkedin.com/in/carmel-christian-wenga-871876178/)