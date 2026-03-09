# Use an NVIDIA CUDA base image that includes development libraries
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    cmake \
    unzip \
    build-essential \
    libgl1-mesa-dev \
    libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev \
    libglib2.0-0

# Install tiny-cuda-nn
# The architectures may need to change depending on your system
ENV TCNN_CUDA_ARCHITECTURES="90;89;86;80"
RUN pip install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn.git@b3473c81396fe927293bdfd5a6be32df8769927c#subdirectory=bindings/torch"

# Install Nerfstudio from pip
RUN pip install nerfstudio

# For Tab completion
RUN ns-install-cli