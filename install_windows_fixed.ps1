# Voice Changer Installation Script for Windows (Fixed Version)
# Run as Administrator for best results

Write-Host "🎤🎵 Voice Changer Installation Script (Fixed)" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if Python is installed
Write-Host "`n🔍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+ from https://python.org" -ForegroundColor Red
    Write-Host "   Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Red
    exit 1
}

# Check if pip is available
Write-Host "`n🔍 Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found. Please reinstall Python with pip" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Check for Visual C++ Build Tools
Write-Host "`n🔧 Checking Visual C++ Build Tools..." -ForegroundColor Yellow
$vcRedist = Get-ItemProperty HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\* | Where-Object {$_.DisplayName -like "*Visual C++*"}
if ($vcRedist) {
    Write-Host "✅ Visual C++ Redistributable found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Visual C++ Redistributable not found" -ForegroundColor Yellow
    Write-Host "   Installing Visual C++ Build Tools..." -ForegroundColor Yellow
    Write-Host "   Please download and install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
    Write-Host "   Install 'C++ build tools' workload" -ForegroundColor Yellow
    Read-Host "Press Enter after installing Visual C++ Build Tools"
}

# Install FFmpeg
Write-Host "`n🎵 Installing FFmpeg..." -ForegroundColor Yellow
try {
    # Try winget first
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Write-Host "✅ FFmpeg installed via winget" -ForegroundColor Green
} catch {
    try {
        # Try Chocolatey
        choco install ffmpeg -y
        Write-Host "✅ FFmpeg installed via Chocolatey" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  FFmpeg installation failed via package managers" -ForegroundColor Yellow
        Write-Host "   Please download manually from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
        Write-Host "   Extract to C:\ffmpeg and add to PATH" -ForegroundColor Yellow
        Read-Host "Press Enter after installing FFmpeg manually"
    }
}

# Install pipwin for PyAudio
Write-Host "`n📦 Installing pipwin..." -ForegroundColor Yellow
pip install pipwin

# Install core packages
Write-Host "`n📦 Installing core packages..." -ForegroundColor Yellow

$packages = @(
    "numpy",
    "requests", 
    "python-dotenv",
    "websockets",
    "soundfile",
    "scipy",
    "pydub"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
}

# Install PyTorch (CPU version)
Write-Host "`n🧠 Installing PyTorch (CPU version)..." -ForegroundColor Yellow
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyAudio
Write-Host "`n🎤 Installing PyAudio..." -ForegroundColor Yellow
try {
    pipwin install pyaudio
    Write-Host "✅ PyAudio installed via pipwin" -ForegroundColor Green
} catch {
    Write-Host "⚠️  PyAudio installation via pipwin failed" -ForegroundColor Yellow
    Write-Host "   Trying alternative method..." -ForegroundColor Yellow
    pip install --only-binary=all pyaudio
}

# Install webrtcvad with pre-compiled wheel
Write-Host "`n🎤 Installing webrtcvad..." -ForegroundColor Yellow
try {
    # Try to install pre-compiled wheel first
    pip install --only-binary=all webrtcvad
    Write-Host "✅ webrtcvad installed via pre-compiled wheel" -ForegroundColor Green
} catch {
    Write-Host "⚠️  webrtcvad pre-compiled wheel not available" -ForegroundColor Yellow
    Write-Host "   Trying to compile from source..." -ForegroundColor Yellow
    try {
        pip install webrtcvad
        Write-Host "✅ webrtcvad compiled successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ webrtcvad compilation failed" -ForegroundColor Red
        Write-Host "   This is required for voice activity detection" -ForegroundColor Red
        Write-Host "   Please ensure Visual C++ Build Tools are installed" -ForegroundColor Red
        Write-Host "   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Red
        Read-Host "Press Enter after installing Visual C++ Build Tools, then run this script again"
        exit 1
    }
}

# Install KojaB libraries with specific versions
Write-Host "`n📚 Installing KojaB libraries..." -ForegroundColor Yellow
try {
    pip install "RealtimeSTT>=0.3.104"
    Write-Host "✅ RealtimeSTT installed" -ForegroundColor Green
} catch {
    Write-Host "❌ RealtimeSTT installation failed" -ForegroundColor Red
    Write-Host "   Trying specific version..." -ForegroundColor Yellow
    pip install "RealtimeSTT==0.3.104"
}

try {
    pip install "RealtimeTTS>=0.3.104"
    Write-Host "✅ RealtimeTTS installed" -ForegroundColor Green
} catch {
    Write-Host "❌ RealtimeTTS installation failed" -ForegroundColor Red
    Write-Host "   Trying specific version..." -ForegroundColor Yellow
    pip install "RealtimeTTS==0.3.104"
}

# Create .env file template
Write-Host "`n🔑 Creating .env file template..." -ForegroundColor Yellow
$envContent = @"
# Voice Changer Configuration
# Replace with your actual API keys

# ElevenLabs API Key (get from https://elevenlabs.io/settings)
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_key_here

# Google Cloud credentials (optional, for alternative STT)
# GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
"@

$envContent | Out-File -FilePath ".env" -Encoding UTF8
Write-Host "✅ .env file created" -ForegroundColor Green

# Test installations
Write-Host "`n🧪 Testing installations..." -ForegroundColor Yellow

# Test Python packages
Write-Host "Testing Python packages..." -ForegroundColor Cyan
try {
    python -c "import numpy; import requests; import pyaudio; import pydub; print('✅ Core packages working')"
} catch {
    Write-Host "❌ Core packages test failed" -ForegroundColor Red
}

# Test webrtcvad
Write-Host "Testing webrtcvad..." -ForegroundColor Cyan
try {
    python -c "import webrtcvad; print('✅ webrtcvad working')"
} catch {
    Write-Host "❌ webrtcvad test failed" -ForegroundColor Red
}

# Test KojaB STT
Write-Host "Testing KojaB STT..." -ForegroundColor Cyan
try {
    python -c "from RealtimeSTT import AudioToTextRecorder; print('✅ KojaB STT working')"
} catch {
    Write-Host "❌ KojaB STT test failed" -ForegroundColor Red
}

# Test KojaB TTS
Write-Host "Testing KojaB TTS..." -ForegroundColor Cyan
try {
    python -c "from RealtimeTTS import TextToAudioStream; print('✅ KojaB TTS working')"
} catch {
    Write-Host "❌ KojaB TTS test failed" -ForegroundColor Red
}

Write-Host "`n🎉 Installation completed!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env file with your API keys" -ForegroundColor White
Write-Host "2. Run: python realtime_voice_changer_koja_hybrid.py" -ForegroundColor White
Write-Host "3. Speak into your microphone!" -ForegroundColor White

Read-Host "Press Enter to exit" 