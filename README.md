# Building Visual Recommendation Systems & Search Engines with Deep Learning, LLMs and Vector Databases

![image](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*BWisOFmpgid3NwahHne-RQ.png)

This repository shows how to build Image-Based Recommendation Systems (IBRS) using different approaches.

From basics implementations to advanced techniques leveraging deep learning, LLM models and vector search engines, in
self-hosted or managed cloud services.

This project is structured into multiple directories, each focusing on a specific method or technology for building IBRS.
* [01 - GoogleNet and MNSLIB](./01%20-%20GoogLeNet%20and%20MNSLIB/README.md): Basic implementations of image-based 
recommendation systems using the GoogleNet model and MNSLIB for similarity search.
* [02 - ResNet and Elasticsearch](./02%20-%20ResNet%20and%20Elasticsearch/README.md): Building IBRS using ResNet for 
embedding extraction and Elasticsearch as vector database. This directory also introduces image-based search capabilities 
and ways to improve recommendation quality by applying image classification using the Gemini Multimodal API.
* [03 - Image Embeddings with VLM-based models](./03%20-%20Image%20Embeddings%20with%20VLM%20based%20models/README.md): 
Improving the quality of image embeddings by using VLM-based models instead of ResNet. The directory explores two 
different strategies to improve the quality of image embeddings: self-hosted model using the SigLIP model locally and 
using the Gemini Multimodal Embedding API via Google AI Studio. Both extracted embeddings are stored in Elasticsearch along with the 
ResNet embeddings, and the recommendation quality of the different approaches is compared.

# Medium Stories Related to this GitHub Repository
1. [Building a Basic Image-Based Recommendation System](https://medium.com/codeelevation/building-a-basic-image-based-recommendation-system-ba9f08588df1)
2. [Building an Image-Based Recommendation System and Search Engine with Deep Learning and Elasticsearch](https://medium.com/towards-artificial-intelligence/building-an-image-based-recommendation-system-and-search-engine-with-deep-learning-and-4bb96c4d9a64)
3. [Building a Smarter Image Search with Gemini and Elasticsearch](https://medium.com/towards-artificial-intelligence/building-a-smarter-image-search-with-gemini-and-elasticsearch-90da55e907b1) 
4. [From ANN Libraries to Vector Databases](https://medium.com/towards-artificial-intelligence/from-ann-libraries-to-vector-databases-06ccda6d918b)
5. [Improving Visual Recommendations with Vision-Language Model Embeddings](https://medium.com/towards-artificial-intelligence/improving-visual-recommendation-with-vision-language-model-embeddings-f299dc744b23)


# Author
Carmel WENGA, Data & ML Engineer
- [Medium](https://medium.com/@carmelwenga)
- [LinkedIn](https://www.linkedin.com/in/carmel-christian-wenga-871876178/)