# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y curl python3 python3-pip git libgl1-mesa-glx libglib2.0-0 unzip && \
    rm -rf /var/lib/apt/lists/*

# Install py-solc-x and Solidity compiler
RUN pip3 install py-solc-x==1.1.0 && python3 -c "from solcx import install_solc; install_solc(version='0.8.20')"

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Download the smart contract zip file and unzip to /app/dataset/contracts
RUN curl -L -o contracts.zip https://github.com/Lilypad-Tech/lilypad-module-synthetic-data/raw/main/refdata/contracts.zip && \
    mkdir -p /app/dataset/contracts && \
    unzip contracts.zip -d /app/dataset/contracts && \
    rm contracts.zip

# Download model
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='xiaodongguaAIGC/llama-3-debug', local_dir='/app/model')"

# Copy application files
COPY Dockerfile /app/Dockerfile
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
COPY start.sh /app/start.sh

RUN ls -la /app

# Make start.sh executable
RUN chmod +x /app/start.sh

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
