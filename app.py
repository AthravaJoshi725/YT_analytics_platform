from fastapi import FastAPI
from fastapi import BackgroundTasks

from contextlib import asynccontextmanager
from collections import Counter

from services.yt_comments import func_get_comments, extract_video_id, extract_video_detail
from services.rag import create_rag, run_rag
from cachetools import TTLCache

import config
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_PATHS['log_file']),
        logging.StreamHandler()
    ]
)



# cache for analysis results upto 10 videos for 30 minutes
# analysis_cache = TTLCache(maxsize=10, ttl=1800)
rag_cache = TTLCache(maxsize=5, ttl=1800)



app = FastAPI(title="Youtube Comment Analyzer")



def run_rag_background(comments, video_id):
    '''
    Create Rag db in background
    Inputs: Comments as list and video_id as str
    Returns: vector db object
    '''
    db = create_rag(comments)

    rag_cache[video_id] = db
    logging.info(f"RAG background task completed -  db saved in cache for {video_id}")



@app.post("/get_comments/")
async def get_comments(youtube_link: str):
    video_id = extract_video_id(youtube_link)
    comments_data = func_get_comments(video_id)
    return {"total_comments": len(comments_data), "comments": comments_data.to_dict(orient="records")}


@app.post("/analyze")
async def analyze(youtube_link: str):
    """
    video_details = {
            "title": item.get("title"),
            "channelName": item.get("channelTitle"),
            "description": item.get("description"),
            "publishedAt": item.get("publishedAt"),
            # dimension of thumnbnail w:1280 h:720
            "thumbnail": item['thumbnails']['maxres'].get("url")
            }
    """


    # get video_id
    video_id = extract_video_id(youtube_link)

    # extract video details
    video_details = extract_video_detail(video_id)

    # if not extract comments and store in dataframe
    df = func_get_comments(video_id)
    if df is None or 'comment' not in df.columns:
        logging.error(f"Failed to extract comments for videoID: {video_id}")
        return {"error": "No comments found or invalid format"}

    # convert the single comment column to list
    comments_data = df['comment'].tolist()

    # storing rag db in cache
    if video_id not in rag_cache:
        db = create_rag(comments_data)
        rag_cache[video_id] = db
        logging.info(f"Rag task completed  and saved in cache for {video_id}")
    
    return video_details


@app.post("/ask")
async def ask_question(youtube_link: str, user_query: str):
    video_id = extract_video_id(youtube_link)
    if video_id not in rag_cache:
        return {"error": "Rag not ready yet. Try again after a few seconds."}
    
    db = rag_cache[video_id]
    rag_response = run_rag(video_id, user_query, db, 10)
    return rag_response['answer']

# myenv\Scripts\python.exe -m uvicorn app:app --reload     