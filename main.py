from fastapi import FastAPI
from fastapi import BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from collections import Counter

from services.yt_comments import func_get_comments, extract_video_id
from services.analysis import get_analysis, check_nltk_data
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


class CommentsRequest(BaseModel):
    youtube_link: str


# cache for analysis results upto 10 videos for 30 minutes
analysis_cache = TTLCache(maxsize=10, ttl=1800)
rag_cache = TTLCache(maxsize=5, ttl=1800)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    check_nltk_data()
    logging.info("nltk data check completed.")

    yield  

app = FastAPI(lifespan=lifespan, title="Youtube Comment Analyzer")



def run_rag_background(comments, video_id):
    '''
    Create Rag db in background
    Inputs: Comments as list and video_id as str
    Returns: vector db object
    '''
    db = create_rag(comments)

    rag_cache[video_id] = db
    logging.info(f"RAG background task completed -  db saved in cache for {video_id}")



def get_percentage(labels):
    '''
    labels : list 
    return : dict {label: percentage} 
    '''
    counts = Counter(label for label in labels)
    total = sum(counts.values())
    percentages = {k: round(v / total * 100, 2) for k, v in counts.items()}
    return percentages


@app.post("/get_comments/")
async def get_comments(req: CommentsRequest):
    video_id = extract_video_id(req.youtube_link)
    comments_data = func_get_comments(video_id)
    return {"total_comments": len(comments_data), "comments": comments_data.to_dict(orient="records")}



@app.post("/analyze")
async def analyze(req: CommentsRequest, background_tasks: BackgroundTasks):
    # get video_id
    video_id = extract_video_id(req.youtube_link)

    # fetch analysis from cache
    if video_id in analysis_cache:
        logging.info(f'Analysis extracted from cache for {video_id}')
        return analysis_cache[video_id]

    # if not extract comments and store in dataframe
    df = func_get_comments(video_id)

    if df is None or 'comment' not in df.columns:
        logging.error(f"Failed to extract comments for videoID: {video_id}")
        return {"error": "No comments found or invalid format"}

    # convert the single comment column to list
    comments_data = df['comment'].tolist()

    # get analysis
    sentiment_result, emotion_result, spam_result, adjectives = get_analysis(comments_data)
    sentiment_distribution = get_percentage(sentiment_result)
    emotion_distribution = get_percentage(emotion_result)
    spam_distribution = get_percentage(spam_result)

    # get adjectives for word cloud

    # add to dataframe
    # df['sentiment'] = sentiment_result
    # df['emotion'] = emotion_result
    # df['spam'] = spam_result

    analysis_result  = {
        "Sentiment": sentiment_distribution,
        "Emotion": emotion_distribution,
        "Spam_detection": spam_distribution,
        "Adjectives": adjectives
    }

    analysis_cache[video_id] = analysis_result
    logging.info(f'Analysis stored in cache for videoID: {video_id}')
    logging.info(f'Analysis completed for videoID: {video_id}')

    # storing rag db in cache
    if video_id not in rag_cache:
        background_tasks.add_task(run_rag_background, comments_data, video_id)
        logging.info(f"Rag background task started for {video_id}")

    return analysis_result



@app.post("/ask")
async def ask_question(req: CommentsRequest, user_query: str):
    video_id = extract_video_id(req.youtube_link)
    if video_id not in rag_cache:
        return {"error": "Rag not ready yet. Try again after a few seconds."}
    
    db = rag_cache[video_id]
    rag_response = run_rag(video_id, user_query, db, 10)
    return rag_response['answer']

# myenv\Scripts\python.exe -m uvicorn app:app --reload     