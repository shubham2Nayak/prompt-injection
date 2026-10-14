@echo off
REM PromptTrace - Startup Script for Windows
REM Launches both backend and frontend

echo.
echo 🚀 PromptTrace Startup Script
echo ==============================
echo.

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do (
    echo ✅ Python found: %%i
)

echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing/updating dependencies...
echo.

REM Install backend requirements
if exist "backend\requirements.txt" (
    echo 📥 Installing backend dependencies...
    pip install -q -r backend\requirements.txt
)

REM Install frontend requirements
if exist "requirements_frontend.txt" (
    echo 📥 Installing frontend dependencies...
    pip install -q -r requirements_frontend.txt
)

echo.
echo ==============================
echo ✅ Setup complete!
echo ==============================
echo.

echo What would you like to do?
echo.
echo 1 - Start Backend only
echo 2 - Start Frontend only
echo 3 - Start Both
echo 4 - Exit
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo 🚀 Starting Backend Server...
    cd backend
    python app.py
) else if "%choice%"=="2" (
    echo 🚀 Starting Frontend Dashboard...
    streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    echo 🚀 Starting Both Services...
    echo.
    echo Starting Backend in new window...
    start cmd /k "cd backend && python app.py"
    timeout /t 2 /nobreak
    
    echo Starting Frontend...
    streamlit run streamlit_app.py
) else if "%choice%"=="4" (
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo ❌ Invalid choice
    exit /b 1
)
