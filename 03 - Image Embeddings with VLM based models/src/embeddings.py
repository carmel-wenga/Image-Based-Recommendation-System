from abc import ABC, abstractmethod
from typing import List
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage


class ImageEmbeddingExtractor(ABC):
    @abstractmethod
    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Extract embeddings for a batch of image paths."""
        pass

class SigLIPEmbeddingExtractor(ImageEmbeddingExtractor):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings.pooler_output
        return embeddings.detach().cpu().numpy()

class VertexAIEmbeddingExtractor(ImageEmbeddingExtractor):
    def __init__(self, model_name="multimodalembedding"):
        self.model = MultiModalEmbeddingModel.from_pretrained(model_name)

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        embeddings = []
        for path in image_paths:
            vertex_image = VertexImage.load_from_file(path)
            emb = self.model.get_embeddings(image=vertex_image)
            embeddings.append(emb.image_embedding)
        return np.array(embeddings)