#! /bin/bash

set -e

# Activate the virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "✅ Virtual environment activated."

# Installing packages
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
fi

# Install Playwright browsers (only once)
echo "🧩 Installing Playwright browser dependencies..."
python -m playwright install

# Start FastAPI backend in background
echo "🌐 Starting FastAPI server..."
uvicorn main:app --reload &  # <- & runs in background
FASTAPI_PID=$!                 # <- saves the process id

# Wait a few seconds for the server to start
sleep 3

# Start Streamlit frontend
echo "🖥️ Starting Streamlit app..."
streamlit run app.py

# Cleanup when Streamlit exits
trap "echo '🛑 Stopping FastAPI server...';
      kill $FASTAPI_PID" EXIT

