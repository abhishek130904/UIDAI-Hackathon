#!/bin/bash

# UIDAI Hackathon - Execution Script
# Team: UIDAI_8037

echo "Starting setup..."

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate and Install
echo "Activating environment and installing requirements..."
source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Fallback if requirements.txt is missing
    pip install pandas numpy matplotlib seaborn geopandas scikit-learn statsmodels
fi

# 4. Run the Analysis
echo "Running visualization dashboard..."
python3 visualization_dashboard.py

echo "Done. Check the 'visualizations' folder for outputs."