# Use an official PyTorch CUDA base image (Python 3.10 + CUDA 12.1 + cuDNN)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /src

# Install system dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & install build backend
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Install the package (uses pyproject.toml) saves some time at builds
# By default: minimal install. For extras, use `--extras viz,dev`
COPY . .
RUN pip install --no-cache-dir .

# Default command (can be overridden at runtime)
CMD ["python"]
