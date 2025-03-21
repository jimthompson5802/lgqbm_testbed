# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.8

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev

# Clone and install LightGBM
# RUN git clone --recursive https://github.com/microsoft/LightGBM && \
#     cd LightGBM && \
#     mkdir build && \
#     cd build && \
#     cmake .. && \
#     make -j4 && \
#     cd ../python-package && \
#     python setup.py install



RUN pip install --upgrade pip setuptools wheel \
    && pip install lightgbm scipy scikit-learn==1.1.1 pandas jupyter ipykernel