@echo off
SETLOCAL ENABLEEXTENSIONS

REM Check if virtual environment exists
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
echo ✅ Virtual environment activated.

REM Install packages
IF EXIST requirements.txt (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
)

REM Install Playwright browser dependencies
echo 🧩 Installing Playwright browser drivers...
python -m playwright install

REM Start FastAPI backend
echo 🌐 Starting FastAPI server...
start "FastAPI Server" cmd /c "uvicorn main:app --reload"

REM Wait a bit for FastAPI to start
timeout /t 3 /nobreak > NUL

REM Start Streamlit app
echo 🖥️ Starting Streamlit app...
streamlit run app.py
