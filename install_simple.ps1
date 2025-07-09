# Simple Voice Changer Installation Script for Windows
# Run as Administrator for best results

Write-Host "🎤🎵 Simple Voice Changer Installation" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

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

# Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install FFmpeg
Write-Host "`n🎵 Installing FFmpeg..." -ForegroundColor Yellow
try {
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Write-Host "✅ FFmpeg installed via winget" -ForegroundColor Green
} catch {
    Write-Host "⚠️  FFmpeg installation failed. Please install manually from https://ffmpeg.org/download.html" -ForegroundColor Yellow
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

Write-Host "`n🎉 Simple installation completed!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env file with your API keys" -ForegroundColor White
Write-Host "2. Run: python realtime_voice_changer_koja_hybrid.py" -ForegroundColor White
Write-Host "3. Speak into your microphone!" -ForegroundColor White
Write-Host "`nNote: If you need webrtcvad for voice activity detection," -ForegroundColor Yellow
Write-Host "   install Visual C++ Build Tools and run: pip install webrtcvad" -ForegroundColor Yellow

Read-Host "Press Enter to exit" 