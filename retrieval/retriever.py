import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class SimpleRetriever:
    def __init__(self):
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.passages = []

    def build_index(self, passages):
        print(f"Building FAISS index for {len(passages)} passages...")
        self.passages = passages
        embeddings = self.model.encode(passages, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve_distractor(self, query, top_k=5):
        """
        Retrieves a semantically similar but incorrect context (distractor).
        We fetch top_k and pick one that is not the exact match but similar.
        """
        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        
        # In a real scenario, we'd ensure it's not the actual gold passage.
        # For simplicity, just return the second best match as a distractor.
        if len(indices[0]) > 1:
            distractor_idx = indices[0][1]
            return self.passages[distractor_idx]
        return self.passages[indices[0][0]]

# This file is currently supplementary, as `dataset_preparation.py` 
# handles the noise generation for the simplified local pipeline.