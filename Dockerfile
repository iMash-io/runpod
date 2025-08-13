# CUDA 12.1 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (cu121) — matches LivePortrait guidance
RUN python3 -m pip install --upgrade pip
RUN pip install "torch==2.3.0+cu121" "torchvision==0.18.0" "torchaudio==2.3.0" \
    --index-url https://download.pytorch.org/whl/cu121

# Core libs
RUN pip install runpod==1.6.2 opencv-python-headless==4.10.0.84 numpy==1.26.4 Pillow==10.4.0
RUN pip install "livekit==1.0.12" "livekit-agents==1.2.5"

# Clone LivePortrait
WORKDIR /app
RUN git clone --depth=1 https://github.com/KwaiVGI/LivePortrait
WORKDIR /app/LivePortrait
RUN pip install -r requirements.txt

# (Optional) FasterLivePortrait TensorRT/ONNX fork — guard with ARG
ARG ENABLE_TRT=false
WORKDIR /app
RUN if [ "$ENABLE_TRT" = "true" ]; then \
      git clone --depth=1 https://github.com/warmshao/FasterLivePortrait && \
      cd FasterLivePortrait && \
      pip install -r requirements.txt || true ; \
    fi

# Weights for LivePortrait
WORKDIR /app/LivePortrait
RUN pip install "huggingface_hub[cli]" && \
    huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"

# App code
WORKDIR /app
COPY app/ app/
COPY rp_handler.py rp_handler.py

# RunPod handler
CMD ["python3", "rp_handler.py"]
