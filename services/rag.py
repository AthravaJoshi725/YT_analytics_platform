from vector_db.vectordb import VectorDB
from services.yt_comments import extract_video_detail

from tenacity import retry, stop_after_attempt, wait_exponential
import config
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
#  Logging Setup 

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

EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, batch_size=8)
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

@retry(stop = stop_after_attempt(3), wait = wait_exponential(multiplier=1, min=2, max=4), reraise=True)
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


def rag_prompt(video_id, user_query, search_results):
    video_details = extract_video_detail(video_id)

    context = "\n\n".join([r["chunk"] for r in search_results])

    return f"""
You are an AI assistant answering strictly based on YouTube comments below.

CONTEXT:
{context}

QUESTION:
{user_query}

YT_VIDEO details:
Video Title: {video_details.get('title', "N/A")}
Channel: {video_details.get('channel_Name', "N/A")}
Uploaded On: {video_details.get('publishedAt', "N/A")}
Description: {video_details.get('description', "N/A")}

INSTRUCTIONS:
- Use ONLY the context.
- Do NOT hallucinate.
- Give a short and clear response.
""".strip()


def embed_query(query: str):
    clean_q = preprocess_comments(query)
    return EMBEDDING_MODEL.encode([clean_q], convert_to_numpy=True)[0]



def run_rag(video_id, user_query, vector_db, embedding_model=EMBEDDING_MODEL, top_k=5):
    start = time.time()
    query_emb = embed_query(user_query)
    retrieved_chunks = vector_db.search(query_emb, top_k )
    prompt = rag_prompt(video_id, user_query, retrieved_chunks)
    answer = generate_llm_response(prompt)
    end = time.time()

    logger.info(f"Full RAG pipeline executed in {end - start:.2f}s")
    # logger.info(f"RAG answer: {retrieved}")
    return {"answer": answer, "chunks_used": [c["chunk"] for c in retrieved_chunks]}


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
