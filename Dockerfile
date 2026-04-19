# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies needed for CatBoost/XGBoost/Sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create artifacts directory for saved models
RUN mkdir -p artifacts

# Expose necessary ports (8501 for Streamlit)
EXPOSE 8501

# Default command
# To run training: docker run <image> python train.py
# To run dashboard: docker run -p 8501:8501 <image>
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
