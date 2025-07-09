@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🎤🎵 Voice Changer Installation Script
echo =====================================

REM Check if Python is installed
echo.
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.11+ from https://python.org
    echo    Make sure to check 'Add Python to PATH' during installation
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo ✅ Python found: !PYTHON_VERSION!
)

REM Check if pip is available
echo.
echo 🔍 Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip not found. Please reinstall Python with pip
    pause
    exit /b 1
) else (
    echo ✅ pip found
)

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install Visual C++ Build Tools check
echo.
echo 🔧 Checking Visual C++ Build Tools...
echo    If you get build errors, install Visual C++ Build Tools from:
echo    https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo    Install 'C++ build tools' workload

REM Install FFmpeg
echo.
echo 🎵 Installing FFmpeg...
winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ FFmpeg installed via winget
) else (
    echo ⚠️  FFmpeg installation via winget failed
    echo    Please download manually from: https://ffmpeg.org/download.html
    echo    Extract to C:\ffmpeg and add to PATH
    pause
)

REM Install pipwin for PyAudio
echo.
echo 📦 Installing pipwin...
pip install pipwin

REM Install core packages
echo.
echo 📦 Installing core packages...
pip install numpy
pip install requests
pip install python-dotenv
pip install websockets
pip install soundfile
pip install scipy

REM Install PyTorch (CPU version)
echo.
echo 🧠 Installing PyTorch (CPU version)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install PyAudio
echo.
echo 🎤 Installing PyAudio...
pipwin install pyaudio >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  PyAudio installation via pipwin failed
    echo    Trying alternative method...
    pip install --only-binary=all pyaudio
) else (
    echo ✅ PyAudio installed via pipwin
)

REM Install pydub
echo.
echo 🎵 Installing pydub...
pip install pydub

REM Install KojaB libraries
echo.
echo 📚 Installing KojaB libraries...
pip install "RealtimeSTT>=0.3.104"
pip install "RealtimeTTS>=0.3.104"

REM Create .env file template
echo.
echo 🔑 Creating .env file template...
(
echo # Voice Changer Configuration
echo # Replace with your actual API keys
echo.
echo # ElevenLabs API Key (get from https://elevenlabs.io/settings)
echo ELEVENLABS_API_KEY=your_elevenlabs_key_here
echo.
echo # OpenAI API Key (get from https://platform.openai.com/api-keys)
echo OPENAI_API_KEY=your_openai_key_here
echo.
echo # Google Cloud credentials (optional, for alternative STT)
echo # GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
) > .env
echo ✅ .env file created

REM Test installations
echo.
echo 🧪 Testing installations...

echo Testing Python packages...
python -c "import numpy; import requests; import pyaudio; import pydub; print('✅ Core packages working')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Core packages working
) else (
    echo ❌ Core packages test failed
)

echo Testing KojaB STT...
python -c "from RealtimeSTT import AudioToTextRecorder; print('✅ KojaB STT working')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ KojaB STT working
) else (
    echo ❌ KojaB STT test failed
)

echo Testing KojaB TTS...
python -c "from RealtimeTTS import TextToAudioStream; print('✅ KojaB TTS working')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ KojaB TTS working
) else (
    echo ❌ KojaB TTS test failed
)

echo.
echo 🎉 Installation completed successfully!
echo =====================================
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: python realtime_voice_changer_koja_hybrid.py
echo 3. Speak into your microphone!
echo.
pause 