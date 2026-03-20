# Improving Visual Recommendations by Improving Image Embeddings Quality

In part two, we saw how to improve the recommendation quality by using Elasticsearch Filtered kNN.

In this third part, we will focus on improving the quality of image embeddings by using VLM-based models instead of 
ResNet. The goal here is to explor two different strategies to improve the quality of image embeddings:

1. Self-hosted model: using the SigLIP model locally.
2. Embedding API: using the Vertex AI Multimodal Embedding API.

Both extracted embeddings will be stored in Elasticsearch along with the ResNet embeddings. We will then compare the 
recommendation quality of the different approaches.

We will also explore the impact of combining text and image embeddings on the recommendation quality.

## Project Setup Instructions
```shell
mkdir etc
python3 -m venv etc/venv-03
source etc/venv-03/bin/activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

## Jupyter Lab Setup
1. Configure jupyter to use the virtual environment that you have created as kernel
```sh
python -m ipykernel install --user --name IBRS-03 --display-name "Python3.12 (IBRS-03)"
```
2. Launch Jupyter Lab with the command below and choose ```Python3.12 (IBRS-03)``` as the kernel
```sh
jupyter lab
```