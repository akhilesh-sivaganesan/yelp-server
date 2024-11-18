#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Check if conda is already installed
if command -v conda &>/dev/null; then
    echo "Conda is already installed. Skipping Miniconda installation."
else
    echo "Conda not found. Installing Miniconda..."
    
    # Download Miniconda installer for Linux (change URL for other OS)
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh
    
    # Make the installer executable
    chmod +x Miniconda3.sh
    
    # Install Miniconda silently
    ./Miniconda3.sh -b -p $HOME/miniconda
    
    # Add conda to PATH
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # Initialize conda (to make it available in future terminals)
    conda init bash
    
    echo "Miniconda installed successfully!"
fi

# Step 2: Install RAPIDS dependencies with conda
echo "Installing RAPIDS dependencies with conda..."
conda install -c rapidsai -c nvidia -c conda-forge cuml=23.10 cudf=23.10 python=3.9 cudatoolkit=11.8 -y

# Step 3: Install additional packages with pip
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
    flask_cors \
    gunicorn

# Display success message
echo "All required libraries installed successfully!"
