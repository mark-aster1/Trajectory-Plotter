@echo off
echo ============================================
echo  Yellow Ball Tracker — Windows Build
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Create venv if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install deps
echo Installing dependencies...
pip install -r requirements.txt

:: Build
echo.
echo Building executable...
pyinstaller yellow_ball_tracker.spec --noconfirm

echo.
if exist "dist\YellowBallTracker.exe" (
    echo ============================================
    echo  BUILD SUCCESSFUL
    echo  Output: dist\YellowBallTracker.exe
    echo ============================================
) else (
    echo BUILD FAILED — check the output above.
)

pause
