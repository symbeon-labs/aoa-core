import torch
import numpy as np
from models.ofp_model import OFPExtractor
from vector_db.index import VectorIndex

class OFPMatcher:
    def __init__(self):
        self.model = OFPExtractor()
        self.model.eval()
        self.db = VectorIndex()

    def extract_ofp(self, image_tensor):
        with torch.no_grad():
            embedding = self.model(image_tensor)
        return embedding.numpy()

    def match(self, query_embedding):
        return self.db.search(query_embedding, k=1)
