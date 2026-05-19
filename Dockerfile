FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --break-system-packages \
    numpy \
    cython \
    pycocotools \
    submitit \
    torchvision \
    scipy \
    onnx \
    onnxruntime \
    tqdm \
    opencv-python \
    torchmetrics \
    PyYAML \
    matplotlib \
    seaborn \
    packaging \
    geojson \
    pyproj \
    geopandas \
    git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi

# Keep container running
CMD ["bash"]
