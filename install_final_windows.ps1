# Final Windows Installation Script for Real-time Voice Changer
# This script installs all dependencies and uses webrtcvad-wheels as the VAD solution

Write-Host "=== Final Real-time Voice Changer Installation (Windows) ===" -ForegroundColor Green
Write-Host "Installing all dependencies with webrtcvad-wheels..." -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Upgrade pip and build tools
Write-Host "Upgrading pip and build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install ALL core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
$coreDeps = @(
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
    "python-dotenv",
    "librosa",
    "soundfile",
    "resampy",
    "audioread"
)

foreach ($dep in $coreDeps) {
    Write-Host "Installing $dep..." -ForegroundColor Cyan
    python -m pip install $dep
}

# Install webrtcvad-wheels (pre-compiled alternative to webrtcvad)
Write-Host "Installing webrtcvad-wheels (pre-compiled VAD)..." -ForegroundColor Yellow
python -m pip install webrtcvad-wheels

# Try to install webrtcvad directly (might work now)
Write-Host "Attempting to install webrtcvad..." -ForegroundColor Yellow
try {
    python -m pip install webrtcvad
    Write-Host "✓ webrtcvad installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "⚠ webrtcvad failed, but webrtcvad-wheels is available" -ForegroundColor Yellow
}

# Install additional audio processing libraries
Write-Host "Installing additional audio libraries..." -ForegroundColor Yellow
$audioDeps = @(
    "webrtcvad-wheels",
    "resampy",
    "audioread"
)

foreach ($dep in $audioDeps) {
    Write-Host "Installing $dep..." -ForegroundColor Cyan
    python -m pip install $dep
}

# Create a comprehensive test script
Write-Host "Creating comprehensive test script..." -ForegroundColor Yellow
$testScript = @"
#!/usr/bin/env python3
import sys
import importlib

def test_import(module_name, alias=None):
    try:
        if alias:
            importlib.import_module(alias)
            print(f"✓ {module_name} (as {alias})")
        else:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

print("=== Testing All Imports ===")
modules = [
    ('numpy', None),
    ('sounddevice', None),
    ('soundfile', None),
    ('librosa', None),
    ('openai', None),
    ('elevenlabs', None),
    ('webrtcvad_wheels', 'webrtcvad'),
    ('pyaudio', None),
    ('scipy', None),
    ('pydub', None),
    ('websockets', None),
    ('asyncio_mqtt', None),
    ('requests', None),
    ('python-dotenv', 'dotenv'),
    ('resampy', None),
    ('audioread', None)
]

all_good = True
for module, alias in modules:
    if not test_import(module, alias):
        all_good = False

print("\n=== Testing VAD Functionality ===")
# Test webrtcvad specifically
try:
    import webrtcvad_wheels as webrtcvad
    vad = webrtcvad.Vad(2)
    print("✓ webrtcvad_wheels working")
    
    # Test with sample audio
    import numpy as np
    sample_audio = np.random.randn(1600).astype(np.int16)  # 100ms at 16kHz
    frame_size = 480  # 30ms at 16kHz
    
    for i in range(0, len(sample_audio) - frame_size, frame_size):
        frame = sample_audio[i:i + frame_size]
        if len(frame) == frame_size:
            is_speech = vad.is_speech(frame.tobytes(), 16000)
            print(f"✓ VAD test frame {i//frame_size}: {'speech' if is_speech else 'silence'}")
            break
    
except Exception as e:
    print(f"✗ webrtcvad_wheels test failed: {e}")
    all_good = False

print("\n=== Testing Audio Libraries ===")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"✓ sounddevice: Found {len(devices)} audio devices")
except Exception as e:
    print(f"✗ sounddevice test failed: {e}")
    all_good = False

try:
    import librosa
    print("✓ librosa working")
except Exception as e:
    print(f"✗ librosa test failed: {e}")
    all_good = False

print("\n=== Final Result ===")
if all_good:
    print("✓ ALL TESTS PASSED! Installation is complete and ready to use.")
else:
    print("✗ Some tests failed. Please check the error messages above.")

print("\n=== Available Audio Devices ===")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']} (in: {device['max_inputs']}, out: {device['max_outputs']})")
except:
    print("Could not list audio devices")
"@

$testScript | Out-File -FilePath "test_complete_installation.py" -Encoding UTF8

Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Testing complete installation..." -ForegroundColor Yellow
python test_complete_installation.py

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Set up your API keys in .env file:" -ForegroundColor White
Write-Host "   ELEVENLABS_API_KEY=your_elevenlabs_key" -ForegroundColor Gray
Write-Host "   OPENAI_API_KEY=your_openai_key" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run: python setup_api_key.py" -ForegroundColor White
Write-Host ""
Write-Host "3. Test the voice changer:" -ForegroundColor White
Write-Host "   python realtime_voice_changer_windows.py" -ForegroundColor Gray
Write-Host ""
Write-Host "If all tests passed, your installation is ready!" -ForegroundColor Green 