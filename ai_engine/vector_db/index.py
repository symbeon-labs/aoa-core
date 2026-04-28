import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim=256):
        self.dim = dim
        # IndexFlatIP uses Inner Product (Cosine Similarity if normalized)
        self.index = faiss.IndexFlatIP(dim)
        self.id_map = {}
        self.current_id = 0

    def add_embeddings(self, embeddings: np.ndarray, gtid: str):
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        n = embeddings.shape[0]
        self.index.add(embeddings.astype(np.float32))
        
        for _ in range(n):
            self.id_map[self.current_id] = gtid
            self.current_id += 1

    def search(self, query_embedding: np.ndarray, k=5):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "score": float(scores[0][i]),
                    "gtid": self.id_map.get(idx, "UNKNOWN")
                })
        return results
