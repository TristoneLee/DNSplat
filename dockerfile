# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Configure conda to use Tsinghua mirror
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda config --set show_channel_urls yes

# Configure pip to use Tsinghua mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Create and activate conda environment
RUN conda create -y -n noposplat python=3.10 \
    && echo "source activate noposplat" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Set working directory
WORKDIR /src

# Copy requirements and install dependencies
COPY requirements.txt .
COPY thirdparty/ ./thirdparty/

# Install PyTorch and other dependencies
RUN source activate noposplat && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt 
    
# Copy source code
COPY src/ ./src/

# Compile CUDA kernels for RoPE
RUN source activate noposplat && \
    cd src/model/encoder/backbone/croco/curope/ && \
    python setup.py build_ext --inplace && \
    cd ../../../../../..

COPY pretrained_weights/ ./pretrained_weights/


# Set default command to bash
CMD ["/bin/bash"]