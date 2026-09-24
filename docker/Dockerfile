# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
# Use a PyTorch CUDA base image aligned with pyproject.toml torch pins
FROM pytorch/pytorch:2.7.0-cuda12.1-cudnn9-runtime

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
    libsndfile1 \
    ffmpeg \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & install build backend
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

COPY . .
ARG PIP_EXTRAS="trainer,models"
RUN pip install --no-cache-dir ".[${PIP_EXTRAS}]"

# Default command (can be overridden at runtime)
CMD ["python"]
