from abc import ABC, abstractmethod
from typing import List
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel


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