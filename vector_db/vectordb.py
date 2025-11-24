import numpy as np
import os
import json
import time
import logging

logger = logging.getLogger(__name__)

#  Vector DB 
class VectorDB:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.embeddings = np.empty((0, embedding_dim), dtype="float32")
        self.chunks = []
        self.ids = []
        self.__next_id = 0

    def add(self, chunk: str, embedding: np.ndarray):
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")

        self.chunks.append(chunk)
        embedding = embedding.reshape(1, -1)

        self.embeddings = np.vstack([self.embeddings, embedding])

        new_id = self.__next_id
        self.ids.append(new_id)
        self.__next_id += 1

        return new_id

    def add_all(self, chunk_list, embedding_list):
        embeddings = np.array(embedding_list, dtype="float32")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")

        self.chunks.extend(chunk_list)
        self.embeddings = np.vstack([self.embeddings, embeddings])

        start_id = self.__next_id
        end_id = start_id + len(chunk_list)

        batch_ids = list(range(start_id, end_id))
        self.ids.extend(batch_ids)

        self.__next_id = end_id
        return batch_ids

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray):
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(matrix_norm, query_norm)
        return scores

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError("Query embedding dimension mismatch")

        start = time.time()
        scores = self._cosine_similarity(query_embedding, self.embeddings)
        end = time.time()

        logger.info(f"Similarity search completed in {end - start:.4f}s")

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.ids[idx],
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
            }
            for idx in top_idx
        ]