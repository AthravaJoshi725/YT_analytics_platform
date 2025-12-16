---
title: Yt Rag Backend
emoji: 🐠
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# YouTube RAG Backend 🎥

Backend API for YouTube video RAG (Retrieval-Augmented Generation) system. Analyzes YouTube videos, extracts comments, and provides intelligent Q&A using RAG.

## 🚀 API Endpoints

- `GET /health` - Health check endpoint
- `POST /analyze` - Analyze YouTube video content
- `GET /get_comments` - Retrieve YouTube video comments
- `POST /ask` - Ask questions about analyzed videos (RAG)

## 🔧 Setup

This Space requires environment variables to be configured in the Space settings:

1. Go to **Settings** → **Variables and secrets**
2. Add your required API keys (e.g., OpenAI, YouTube API, etc.)

## 📡 Usage
```bash
# Health check
curl https://aj343-yt-rag-backend.hf.space/health

# Analyze video
curl -X POST https://aj343-yt-rag-backend.hf.space/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_url": "YOUR_YOUTUBE_URL"}'

# Ask questions
curl -X POST https://aj343-yt-rag-backend.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## 🐳 Docker

This Space runs on Docker. The container exposes port 7860.