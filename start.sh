#!/bin/bash

# Step 1: Activate the Conda environment
source $HOME/miniconda/bin/activate myenv

# Step 2: Run Gunicorn to start the Flask application
gunicorn app:app
