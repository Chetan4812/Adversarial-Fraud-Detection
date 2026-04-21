# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    APP_HOME=/app

# Set work directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create artifacts directory and set permissions
RUN mkdir -p artifacts && \
    useradd -m streamlituser && \
    chown -R streamlituser:streamlituser $APP_HOME

# Switch to non-root user
USER streamlituser

# Expose necessary ports
EXPOSE 8501

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py"]
