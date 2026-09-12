# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN pip install uv && uv sync --frozen

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default command to start the API
CMD ["uv", "run", "python", "api/main.py"]
