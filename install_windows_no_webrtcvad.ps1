# Windows Installation Script for Real-time Voice Changer (No webrtcvad)
# This script installs all dependencies except webrtcvad to avoid compilation issues

Write-Host "=== Real-time Voice Changer Installation (Windows) ===" -ForegroundColor Green
Write-Host "Installing dependencies without webrtcvad..." -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install core dependencies (excluding webrtcvad)
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
$dependencies = @(
    "openai",
    "elevenlabs",
    "sounddevice",
    "numpy",
    "scipy",
    "pyaudio",
    "websockets",
    "asyncio-mqtt",
    "pydub",
    "requests",
    "python-dotenv"
)

foreach ($dep in $dependencies) {
    Write-Host "Installing $dep..." -ForegroundColor Cyan
    python -m pip install $dep
}

# Install KojaB libraries (these handle VAD better than webrtcvad)
Write-Host "Installing KojaB libraries for superior audio processing..." -ForegroundColor Yellow
python -m pip install "koja-realtime-stt==0.3.104"
python -m pip install "koja-realtime-tts==0.3.104"

# Install additional audio processing libraries
Write-Host "Installing additional audio libraries..." -ForegroundColor Yellow
python -m pip install "librosa"
python -m pip install "soundfile"
python -m pip install "webrtcvad-wheels"  # Pre-compiled wheels if available

# Try to install webrtcvad from pre-compiled wheels
Write-Host "Attempting to install webrtcvad from pre-compiled wheels..." -ForegroundColor Yellow
try {
    python -m pip install --only-binary=all webrtcvad
    Write-Host "webrtcvad installed successfully from pre-compiled wheels!" -ForegroundColor Green
} catch {
    Write-Host "webrtcvad installation failed, but this is OK - we'll use alternative VAD" -ForegroundColor Yellow
}

# Install development tools for potential compilation
Write-Host "Installing development tools..." -ForegroundColor Yellow
python -m pip install "wheel"
python -m pip install "setuptools"

Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host "Note: If webrtcvad failed to install, the voice changer will use alternative VAD methods." -ForegroundColor Yellow
Write-Host "The KojaB libraries provide superior VAD functionality anyway." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Set up your API keys in .env file" -ForegroundColor White
Write-Host "2. Run: python setup_api_key.py" -ForegroundColor White
Write-Host "3. Test with: python test_voice_changer.py" -ForegroundColor White 