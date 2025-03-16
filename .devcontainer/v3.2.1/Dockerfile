# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11

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
    && pip install lightgbm==3.2.1 scipy scikit-learn pandas jupyter ipykernel