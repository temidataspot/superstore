# Base image 
FROM python:3.11-slim

# Metadata 
LABEL maintainer="Temi Priscilla Jokotola"
LABEL description="Superstore Customer Analytics Pipeline"

#  Environment 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# Working directory
WORKDIR /app

# System dependencies 
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files 
COPY . .

# Create directories 
RUN mkdir -p models outputs

# Default command — run the full pipeline 
CMD ["python", "scheduler.py"]

