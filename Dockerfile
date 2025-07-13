# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for models
RUN mkdir -p models

# Train the model during build
RUN python -m app.services.train_model

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
