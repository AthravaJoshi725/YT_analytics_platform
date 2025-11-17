import pandas as pd
import numpy as np
import os
import json
import re
import logging
import config
import time

from sentence_transformers import SentenceTransformer

log_file = config.OUTPUT_PATHS.get('log_file', 'app.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_PATHS['log_file']),
        logging.StreamHandler()
    ]
)

EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def preprocess_comments(text:str) -> str:
    if not isinstance(text, str):
        return ""
        
    # remove urls
    text = re.sub(r'https\S+|www\.\S+', "", text)

    # lowercase
    text = text.lower()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text 

def chunk_comments(comments, max_words = 200):
    chunks = []
    current_chunk = []
    word_count = 0
    
    for comment in comments:
        words = comment.split()

        if len(words) + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
        current_chunk.append(comment)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logging.info(f"{len(chunks)} chunks successfully creaeted")
    return chunks

def embedding_chunks(chunks, embedding_model):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=32
        )
    logging.INFO(f"{embeddings.shape}")
    return embeddings

class VectorDB:
    def __init__(self,embedding_dim:int):
        '''
        Embedding_dim  = size of each embedding (number of features used to represent the meaning)
        [768 for mpnet]
        '''

        self.embedding_dim = embedding_dim
        self.embeddings = np.empty((0,embedding_dim), dtype='float32')
        self.chunks = [] # text storage
        self.ids = [] # simple incremental IDS

        self.__next_id = 0
    
    def add(self, chunk:str, embedding: np.array):
        """
        Insert a single chunk + embedding 
        

        """

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")
        
        # add text
        self.chunks.append(chunk)

        # add embedding --- shape of input embedding = (768,)
        embedding = embedding.reshape(1,-1)
        # after reshape --- (1,768)

        self.embeddings = np.vstack([self.embeddings, embedding])

        # add ID
        new_id = self.__next_id
        self.ids.append(new_id)

        self.__next_id += 1

        return new_id

    def add_all(self, chunk_list, embedding_list):
        '''
        Insert multiple chunks + embedding in batch 
        '''
        # convert the embedding list to numpy
        embeddings_list = np.array(embedding_list, dtype="float32")

        if embeddings_list.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")
        
        # add all chunks
        self.chunks.extend(chunk_list)
        # add all embeddings at once
        self.embeddings = np.vstack([self.embeddings, embedding_list])

        # create ids for this batch
        start_id = self.__next_id
        end_id = start_id + len(chunk_list)

        batch_ids = list(range(start_id,end_id))
        self.ids.extend(batch_ids)

        self.__next_id = end_id

        return batch_ids
    
    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray):
        """
        Compute similarity between query vector and all stored embeddings.
        """
        # normalize query
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        # normalize all embeddings
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True)+ 1e-10)

        scores = np.dot(matrix_norm, query_norm)

        return scores # shape: (num_chunks, ) a list of num_chunks which is 227
    
    def search(self, query_embedding: np.ndarray, top_k: int=5):
        """
        Find top k most similar chunks to the query embedding
        Return: list of dict{id, chunk, score}
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError("Embedding dimensions mismatch")

        # compute cosine similarity
        scores = self._cosine_similarity(query_embedding, self.embeddings)

        # get top-k indexes in descending order
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_idx:
            results.append({
                "id": self.ids[idx],
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })
        
        return results
    
    def save(self,folder_path:str):
        """
        Save embeddings, chunks and metadata to disk 
        """

        os.makedirs(folder_path, exist_ok=True)

        # save emebddings
        np.save(os.path.join(folder_path, "embeddings.npy"), self.embeddings)

        # save chunks + id + metadata
        metadata = {
            "chunks" : self.chunks,
            "ids" : self.ids,
            "embedding_dim": self.embedding_dim,
            "next_id": self.__next_id
        }

        with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        return True
    
    @classmethod
    def load(cls, folder_path: str):
        """
        load the disk
        """
        
        # load metadata
        with open(os.path.join(folder_path, "metadata.json"), 'r', encoding="utf-8") as f:
            metadata = json.load(f)
        
        # create new instance [ cls == Vectordb]
        db = cls(embedding_dim = metadata['embedding_dim'])

        # load embeddings
        db.embeddings = np.load(os.path.join(folder_path, "embeddings.npy"))

        # load chunks + ids
        db.chunks = metadata['chunks']
        db.ids = metadata['ids']
        db.__next_id = metadata['next_id']

        return db


def rag_result(comments, max_words):
    logging.info("Starting RAG system")

    # clean the comments
    cleaned_comments  = [preprocess_comments(c) for c in comments]
    logging.info(f"{len(cleaned_comments)} comments cleaned")

    # Split the comments into chunks
    start = time.time()
    chunks = chunk_comments(cleaned_comments, max_words=max_words)
    end = time.time()
    logging.info(f"{len(chunks)} chunks created in time: {end - start:.2f}s")

    # Convert the chunks into embeddings using sentence transformers
    start = time.time()
    embeddings = embedding_chunks(chunks, EMBEDDING_MODEL)
    embeddings_dimension = embeddings.shape[1]
    end = time.time()
    logging.info(f"Chunks converted to embeddings in time {end - start:.2f}s")

    # Create a vector database
    db = VectorDB(embeddings_dimension)
    batch_ids = db.add_all(chunks, embeddings)
    
    return db

def search_rag(db,user_query, k):
    """
    This function will take query convert to embeddings and then search
    """

def ai_answer():
    pass