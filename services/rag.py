from vector_db.vectordb import VectorDB

import pandas as pd
import numpy as np
import os
import json
import re
import logging
import time

from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import config
#  Logging Setup 
from pathlib import Path

LOG_FILE = Path("logs/app.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

log_file = LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_PATHS['log_file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


#  Load Models & API 
load_dotenv()
API_KEY = os.environ.get("GENAI_API_KEY", None)

EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
        logger.info("Gemini client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None
else:
    logger.warning("GENAI_API_KEY not found. Gemini responses will fail.")


#  RAG Utilities 
def preprocess_comments(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https\S+|www\.\S+", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_comments(comments, max_words=200):
    chunks, current_chunk = [], []
    word_count = 0

    for comment in comments:
        words = comment.split()

        if len(words) + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0

        current_chunk.append(comment)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


def embedding_chunks(chunks, embedding_model=EMBEDDING_MODEL):
    start = time.time()
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, batch_size=32)
    end = time.time()

    logger.info(
        f"Chunk embeddings created: shape={embeddings.shape}, time={end - start:.2f}s"
    )
    return embeddings


def create_rag(comments, max_words=200):
    logger.info("Initializing RAG pipeline...")

    cleaned_comments = [preprocess_comments(c) for c in comments]
    logger.info(f"Cleaned {len(cleaned_comments)} comments.")

    chunks = chunk_comments(cleaned_comments, max_words=max_words)
    embeddings = embedding_chunks(chunks)

    db = VectorDB(embeddings.shape[1])
    db.add_all(chunks, embeddings)

    logger.info("VectorDB created successfully.")
    return db


def generate_llm_response(prompt):
    if not client:
        return "[ERROR: Gemini client is not initialized]"

    start = time.time()
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
    )
    end = time.time()

    logger.info(f"Gemini response generated in {end - start:.2f}s")
    return resp.text


def rag_prompt(user_query, search_results):
    context = "\n\n".join([r["chunk"] for r in search_results])

    return f"""
You are an AI assistant answering strictly based on YouTube comments below.

CONTEXT:
{context}

QUESTION:
{user_query}

INSTRUCTIONS:
- Use ONLY the context.
- Do NOT hallucinate.
- Give a short and clear response.
""".strip()


def run_rag(user_query, vector_db, embedding_model=EMBEDDING_MODEL, top_k=5):
    start = time.time()
    query_emb = embedding_chunks(user_query)
    retrieved = vector_db.search(query_emb, top_k)
    prompt = rag_prompt(user_query, retrieved)
    answer = generate_llm_response(prompt)
    end = time.time()

    logger.info(f"Full RAG pipeline executed in {end - start:.2f}s")

    return {"answer": answer, "chunks_used": retrieved}


#  TESTING 
if __name__ == "__main__":
    dummy_comments = [
        "This video is amazing!",
        "I really liked the editing.",
        "Worst video ever made.",
        "Music was too loud.",
        "Loved the tutorial, very helpful!",
    ]

    logger.info("Running full RAG test on dummy data...")

    db = create_rag(dummy_comments, max_words=50)

    user_query = "What did people say about the quality?"
    result = run_rag(user_query, db, EMBEDDING_MODEL, top_k=3)

    print("\nANSWER:\n", result["answer"])
    print("\nCHUNKS USED:")
    for c in result["chunks_used"]:
        print(c)
