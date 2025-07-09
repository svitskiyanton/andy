# Complete Windows Installation Script for Real-time Voice Changer
# This script installs ALL dependencies and sets up the environment properly

Write-Host "=== Complete Real-time Voice Changer Installation (Windows) ===" -ForegroundColor Green
Write-Host "Installing ALL dependencies including webrtcvad..." -ForegroundColor Yellow

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

# Install Microsoft Visual C++ Build Tools if not present
Write-Host "Checking for Visual C++ Build Tools..." -ForegroundColor Yellow
try {
    $vcvars = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Name "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vcvars) {
        Write-Host "Visual C++ Build Tools found!" -ForegroundColor Green
    } else {
        Write-Host "Visual C++ Build Tools not found. Please install from:" -ForegroundColor Yellow
        Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Cyan
        Write-Host "After installation, run this script again." -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "Visual C++ Build Tools not found. Please install from:" -ForegroundColor Yellow
    Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Cyan
    Write-Host "After installation, run this script again." -ForegroundColor Yellow
    exit 1
}

# Install core dependencies
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
    "soundfile"
)

foreach ($dep in $coreDeps) {
    Write-Host "Installing $dep..." -ForegroundColor Cyan
    python -m pip install $dep
}

# Install webrtcvad-wheels (pre-compiled alternative)
Write-Host "Installing webrtcvad-wheels..." -ForegroundColor Yellow
python -m pip install webrtcvad-wheels

# Try to install webrtcvad with proper environment
Write-Host "Attempting to install webrtcvad with Visual C++ environment..." -ForegroundColor Yellow

# Find Visual Studio installation
$vsPath = $null
$possiblePaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $vcvarsPath = Join-Path $path "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvarsPath) {
            $vsPath = $path
            break
        }
    }
}

if ($vsPath) {
    Write-Host "Found Visual Studio at: $vsPath" -ForegroundColor Green
    $vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    
    # Set up environment and install webrtcvad
    Write-Host "Setting up Visual C++ environment and installing webrtcvad..." -ForegroundColor Yellow
    try {
        # Create a batch file to run the installation
        $batchContent = @"
@echo off
call "$vcvarsPath"
python -m pip install webrtcvad
"@
        
        $batchFile = "install_webrtcvad.bat"
        $batchContent | Out-File -FilePath $batchFile -Encoding ASCII
        
        # Run the batch file
        & cmd /c $batchFile
        
        # Clean up
        Remove-Item $batchFile -ErrorAction SilentlyContinue
        
        Write-Host "webrtcvad installation completed!" -ForegroundColor Green
    } catch {
        Write-Host "webrtcvad installation failed, but webrtcvad-wheels is available" -ForegroundColor Yellow
    }
} else {
    Write-Host "Visual Studio not found in expected locations" -ForegroundColor Yellow
    Write-Host "Trying alternative webrtcvad installation..." -ForegroundColor Yellow
    python -m pip install webrtcvad
}

# Install additional audio processing libraries
Write-Host "Installing additional audio libraries..." -ForegroundColor Yellow
$audioDeps = @(
    "resampy",
    "audioread",
    "webrtcvad-wheels"
)

foreach ($dep in $audioDeps) {
    Write-Host "Installing $dep..." -ForegroundColor Cyan
    python -m pip install $dep
}

# Create a test script to verify installation
Write-Host "Creating test script..." -ForegroundColor Yellow
$testScript = @"
#!/usr/bin/env python3
import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

print("Testing imports...")
modules = [
    'numpy',
    'sounddevice',
    'soundfile',
    'librosa',
    'openai',
    'elevenlabs',
    'webrtcvad_wheels',
    'pyaudio',
    'scipy'
]

all_good = True
for module in modules:
    if not test_import(module):
        all_good = False

if all_good:
    print("\n✓ All modules imported successfully!")
else:
    print("\n✗ Some modules failed to import")

# Test webrtcvad specifically
try:
    import webrtcvad_wheels as webrtcvad
    vad = webrtcvad.Vad(2)
    print("✓ webrtcvad_wheels working")
except Exception as e:
    print(f"✗ webrtcvad_wheels: {e}")
"@

$testScript | Out-File -FilePath "test_imports.py" -Encoding UTF8

Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Testing installation..." -ForegroundColor Yellow
python test_imports.py

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Set up your API keys in .env file" -ForegroundColor White
Write-Host "2. Run: python setup_api_key.py" -ForegroundColor White
Write-Host "3. Test with: python realtime_voice_changer_windows.py" -ForegroundColor White
Write-Host ""
Write-Host "If any modules failed to import, please check the error messages above." -ForegroundColor Yellow 