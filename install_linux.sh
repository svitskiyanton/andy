#!/bin/bash

echo "Installing Real-Time Voice Changer for Linux/macOS..."
echo "=================================================="

echo
echo "Step 1: Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "Step 2: Installing PyAudio..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux. Installing system dependencies..."
    
    # Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-pyaudio portaudio19-dev
    # CentOS/RHEL/Fedora
    elif command -v yum &> /dev/null; then
        sudo yum install -y portaudio-devel
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y portaudio-devel
    else
        echo "Warning: Could not detect package manager. You may need to install portaudio manually."
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS. Installing system dependencies..."
    
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "Warning: Homebrew not found. Please install portaudio manually:"
        echo "  brew install portaudio"
    fi
fi

# Install PyAudio
pip install pyaudio

if [ $? -ne 0 ]; then
    echo
    echo "=================================================="
    echo "PyAudio installation failed!"
    echo "=================================================="
    echo
    echo "Manual installation options:"
    echo
    echo "Linux (Ubuntu/Debian):"
    echo "  sudo apt-get install python3-pyaudio"
    echo
    echo "macOS:"
    echo "  brew install portaudio"
    echo "  pip install pyaudio"
    echo
    echo "Alternative: Use conda"
    echo "  conda install pyaudio"
    echo
    exit 1
fi

echo
echo "Step 3: Creating environment file..."
if [ ! -f .env ]; then
    cp env_template.txt .env
    echo "Created .env file. Please edit it with your ElevenLabs credentials."
else
    echo ".env file already exists."
fi

echo
echo "=================================================="
echo "Installation completed successfully!"
echo "=================================================="
echo
echo "Next steps:"
echo "1. Edit .env file with your ElevenLabs API key and voice ID"
echo "2. Run: python test_voice_changer.py"
echo "3. Run: python real_time_voice_changer.py"
echo 