# Railway Dockerfile for the Python backend
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080

# Default to WebSocket server, but allow override via Railway
CMD ["python", "websocket_server.py"]