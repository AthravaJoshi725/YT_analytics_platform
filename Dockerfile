FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch (smaller binary)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clean up caches to reduce image size
RUN rm -rf /root/.cache/* && pip cache purge

# Copy application code
COPY . .

# Expose port (Railway will override this with $PORT)
EXPOSE 8000

# CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
