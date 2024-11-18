#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Install RAPIDS dependencies with conda
echo "Installing RAPIDS dependencies with conda..."
conda install -c rapidsai -c nvidia -c conda-forge cuml=23.10 cudf=23.10 python=3.9 cudatoolkit=11.8 -y

# Step 2: Install remaining packages with pip
echo "Installing additional Python packages with pip..."
pip install pickle5 \
    transformers \
    torch \
    numpy \
    tqdm \
    pandas \
    joblib \
    scikit-learn \
    matplotlib \
    flask \
    flask_cors

# Display installation success message
echo "All required libraries installed successfully!"
