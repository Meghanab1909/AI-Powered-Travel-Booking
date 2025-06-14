#!/bin/bash

set -e

# Activate the virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated."

# Start FastAPI backend (localhost:8000 by default)
echo "🌐 Starting FastAPI server..."
uvicorn main:app --reload &
FASTAPI_PID=$!

sleep 3

# Start Streamlit frontend
echo "🖥️ Starting Streamlit app..."
streamlit run app.py

# Clean up
trap "echo '🛑 Stopping FastAPI server...'; kill $FASTAPI_PID" EXIT

