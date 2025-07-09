# 🎤 Manual Installation Guide

## **Prerequisites**

### **1. Python 3.11+**
- Download from: https://python.org
- ✅ Check "Add Python to PATH" during installation
- ✅ Check "Install pip" during installation

### **2. Visual C++ Build Tools**
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "C++ build tools" workload
- Required for PyAudio compilation

### **3. FFmpeg**
```cmd
# Option 1: Using winget (recommended)
winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements

# Option 2: Manual download
# Download from: https://ffmpeg.org/download.html
# Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH
```

## **Step-by-Step Installation**

### **Step 1: Upgrade pip**
```cmd
python -m pip install --upgrade pip
```

### **Step 2: Install pipwin for PyAudio**
```cmd
pip install pipwin
```

### **Step 3: Install core packages**
```cmd
pip install numpy requests python-dotenv websockets soundfile scipy pydub
```

### **Step 4: Install PyTorch (CPU version)**
```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **Step 5: Install PyAudio**
```cmd
pipwin install pyaudio
```

### **Step 6: Install KojaB libraries**
```cmd
pip install "RealtimeSTT>=0.3.104"
pip install "RealtimeTTS>=0.3.104"
```

### **Step 7: Create .env file**
Create a file named `.env` in your project directory:
```env
# ElevenLabs API Key (get from https://elevenlabs.io/settings)
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_key_here
```

## **Test Installation**

```cmd
# Test all components
python -c "import numpy; import requests; import pyaudio; import pydub; from RealtimeSTT import AudioToTextRecorder; from RealtimeTTS import TextToAudioStream; print('✅ All systems working!')"
```

## **Run Voice Changer**

```cmd
python realtime_voice_changer_koja_hybrid.py
```

## **Troubleshooting**

### **PyAudio Installation Fails**
```cmd
# Try alternative method
pip install --only-binary=all pyaudio
```

### **KojaB Libraries Fail**
```cmd
# Install specific versions
pip install RealtimeSTT==0.3.104
pip install RealtimeTTS==0.3.104
```

### **FFmpeg Not Found**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH environment variable
4. Restart your command prompt

### **Build Errors**
1. Install Visual C++ Build Tools
2. Restart your computer
3. Try installation again

## **API Keys Setup**

### **ElevenLabs API Key**
1. Go to https://elevenlabs.io/settings
2. Copy your API key
3. Paste in `.env` file

### **OpenAI API Key**
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Copy and paste in `.env` file

## **Usage**

1. **Edit `.env` file** with your API keys
2. **Run the voice changer:**
   ```cmd
   python realtime_voice_changer_koja_hybrid.py
   ```
3. **Speak into your microphone!** 🎤

The voice changer will:
- Use KojaB's superior audio input with WebRTC + Silero VAD
- Transcribe speech using Whisper Online API
- Generate voice using ElevenLabs Pro Flash v2.5
- Stream audio back in real-time 