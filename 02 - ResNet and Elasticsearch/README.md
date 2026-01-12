# Build an Image-Based Recommendation System with ResNet, Elasticsearch, and Python

## Requirements
* Python 3.12
* Elasticsearch 9.1.8 via Docker
* Elasticsearch Python Client 9.2.0
* TensorFlow 2.16.1

## Project Setup Instructions
```shell
mkdir etc
python3 -m venv etc/venv
source etc/venv/bin/activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

## Run Elasticsearch
Use the following command to run Elasticsearch in a Docker container:

```shell
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:9.1.8
```
The `xpack.security.enabled=false` property is used to disable security for local development. It allows connecting and interacting with 
Elasticsearch with http://localhost:9200 without authentication.

## The Notebook
The tutorial is implemented in the [IBRS_with_ResNet_and_ES.ipynb](./src/IBRS_with_ResNet_and_ES.ipynb) Jupyter Notebook.

## Author
Carmel WENGA, Data & ML Engineer
- [Medium](https://medium.com/@carmelwenga)
- [LinkedIn](https://www.linkedin.com/in/carmel-christian-wenga-871876178/)