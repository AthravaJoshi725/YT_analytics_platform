"""
Configuration file for Community Feedback Analyser
Centralizes all paths and settings
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Folder paths
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Model file paths
MODEL_PATHS = {
    "sentiment_model": MODELS_DIR / "trained_model.sav",
    "sentiment_vectorizer": MODELS_DIR / "tfidf_vectorizer.sav",
    "spam_model": MODELS_DIR / "model_spamDetect.pkl",
    "emotion_model": MODELS_DIR / "emotion_detect.pkl"
}

# Output file paths
OUTPUT_PATHS = {
    "comments_json": OUTPUT_DIR / "comments.json",
    "comments_csv": OUTPUT_DIR / "comments.csv",
    "log_file": LOGS_DIR / "app.log"
}

# Ensure directories exist
for directory in [MODELS_DIR, DATASETS_DIR, OUTPUT_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

