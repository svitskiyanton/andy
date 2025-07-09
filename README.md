# Real-Time Voice Changer with ElevenLabs

A Python application that transforms your voice in real-time using ElevenLabs' Speech-to-Speech API. Speak into your microphone and hear your voice transformed through your speakers instantly!

## Features

- 🎤 **Real-time voice transformation** from microphone to speakers
- 🔄 **Ultra-low latency** with optimized buffering and parallel processing
- 🎯 **Preserves emotion and delivery** of your original voice
- 📊 **Performance monitoring** with live statistics
- 🛑 **Graceful shutdown** with Ctrl+C
- 🔧 **Automatic device detection** (microphone and speakers)
- ⚡ **Two versions available**: Standard and Advanced (parallel processing)
- 🌍 **Multi-language support** with STT→TTS pipeline (Russian, etc.)

## Voice Changer Approaches

### 1. Speech-to-Speech (STS) - Original Approach
- **Best for**: English, high-quality voice transformation
- **Latency**: 200-800ms
- **Quality**: Excellent for English, poor for Russian
- **Files**: `real_time_voice_changer.py`, `real_time_voice_changer_advanced.py`

### 2. STT→TTS Pipeline - New Approach
- **Best for**: Russian, multi-language support, better quality
- **Latency**: 500ms-1.5s (slightly higher but better quality)
- **Quality**: Excellent for all languages including Russian
- **Files**: `realtime_stt_tts_voice_changer.py`
- **Requirements**: Google Cloud Speech-to-Text API

## Quick Start

### Easy Launcher
```bash
python run_voice_changer.py
```
This will let you choose between Standard and Advanced versions.

### Direct Launch
```bash
# Standard version (recommended for most users)
python real_time_voice_changer.py

# Advanced version (lower latency, higher resource usage)
python real_time_voice_changer_advanced.py

# STT→TTS version (best for Russian and multi-language)
python realtime_stt_tts_voice_changer.py
```

## Latency Optimizations

### What Was Fixed
1. **High latency (2-5 seconds)** → **Now 200-800ms average**
2. **No audio output** → **Fixed audio format handling**
3. **Inefficient chunking** → **Optimized buffer sizes**
4. **Sequential processing** → **Parallel API workers (advanced version)**

### Key Improvements
- **Larger audio buffers**: 2-second chunks instead of 64ms chunks
- **Maximum latency optimization**: Level 4 (ElevenLabs setting)
- **Proper audio format handling**: Correct PCM to WAV conversion
- **Parallel processing**: Multiple API workers (advanced version)
- **Ordered playback**: Maintains audio sequence integrity

## Prerequisites

- Python 3.7 or higher
- ElevenLabs API key
- Microphone and speakers
- Internet connection

## Installation

### 1. Clone or download this project

### 2. Install dependencies

#### **Windows:**
```bash
# Option A: Use the automated installer (recommended)
install_windows.bat

# Option B: Manual installation
pip install -r requirements.txt
pip install pipwin
pipwin install pyaudio
```

#### **Linux/macOS:**
```bash
# Option A: Use the automated installer (recommended)
chmod +x install_linux.sh
./install_linux.sh

# Option B: Manual installation
pip install -r requirements.txt

# Ubuntu/Debian:
sudo apt-get install python3-pyaudio portaudio19-dev
pip install pyaudio

# macOS:
brew install portaudio
pip install pyaudio
```

### 3. Setup environment variables
```bash
# Copy the template file
cp env_template.txt .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Add your ElevenLabs credentials to `.env`:
```
ELEVENLABS_API_KEY=your_actual_api_key_here
VOICE_ID=your_target_voice_id_here
```

### 4. Get your ElevenLabs credentials

1. **API Key**: 
   - Go to [ElevenLabs Dashboard](https://elevenlabs.io/)
   - Create an account or sign in
   - Go to Profile → API Key
   - Copy your API key

2. **Voice ID**:
   - Go to Voice Library in your dashboard
   - Choose a voice you want to transform to
   - Copy the voice ID (looks like: `JBFqnCBsd6RMkjVDRZzb`)

### 5. Setup Google Cloud (for STT→TTS version only)

If you want to use the STT→TTS pipeline (recommended for Russian and multi-language support):

```bash
# Run the setup helper
python setup_google_cloud.py
```

Or manually:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Cloud Speech-to-Text API"
4. Go to "APIs & Services" > "Credentials"
5. Create a service account and download the JSON key
6. Add to your `.env` file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   ```

## Usage

### Test Your Setup First

**For Speech-to-Speech (STS) versions:**
```bash
python test_voice_changer.py
```

**For STT→TTS version:**
```bash
# Test the complete pipeline
python test_stt_tts_pipeline.py
```

### Start the Voice Changer
```bash
python run_voice_changer.py
```

### What You'll See:
```
🚀 Starting Real-Time Voice Changer...
============================================================
🎯 Press Ctrl+C to stop
============================================================
🎤 Input Device: Microphone (Realtek Audio)
🔊 Output Device: Speakers (Realtek Audio)
📊 Sample Rate: 16000Hz, Channels: 1
📦 Chunk Size: 4096 samples (256ms)
🔄 Buffer Duration: 2s
✅ Audio setup complete
✅ Started Capture thread
✅ Started Process thread
✅ Started Play thread
✅ Started Monitor thread
📊 Stats: 45 chunks processed, 9.0 chunks/sec, Avg Latency: 350ms
```

### Stop the Voice Changer
Press `Ctrl+C` to stop gracefully.

## Version Comparison

| Feature | Standard Version | Advanced Version |
|---------|------------------|------------------|
| **API Workers** | 1 | 2 (parallel) |
| **Buffer Duration** | 2 seconds | 1 second |
| **Chunk Size** | 4096 samples | 2048 samples |
| **Latency** | ~300-600ms | ~200-400ms |
| **Resource Usage** | Low | Medium |
| **Recommended For** | Most users | Power users |

## How It Works

```
Microphone → Audio Buffer → ElevenLabs API → Speakers
     ↓            ↓              ↓            ↓
   Capture   →  Batch      →  Transform  →  Play
```

1. **Audio Capture**: Captures audio from your microphone in real-time
2. **Buffering**: Collects 1-2 seconds of audio before processing
3. **Processing**: Sends audio to ElevenLabs Speech-to-Speech API
4. **Transformation**: ElevenLabs AI transforms your voice to the target voice
5. **Output**: Plays the transformed audio through your speakers

## Configuration

### Audio Settings
The default settings are optimized for low latency:
- **Sample Rate**: 16kHz
- **Channels**: Mono (1 channel)
- **Format**: 16-bit PCM
- **Standard Chunk Size**: 4096 samples (256ms)
- **Advanced Chunk Size**: 2048 samples (128ms)

### Latency Optimization
The script uses ElevenLabs' maximum latency optimization (level 4), which provides:
- ~85% latency improvement
- Best balance of quality and speed

You can modify the `optimize_streaming_latency` parameter in the code:
- `0`: Best quality, highest latency
- `1`: Normal optimizations
- `2`: Strong optimizations
- `3`: Maximum optimizations
- `4`: Maximum + text normalizer off (current setting)

## Troubleshooting

### Common Issues

#### 1. "No module named 'pyaudio'"

**Windows:**
```bash
# Option 1: Use pipwin (recommended)
pip install pipwin
pipwin install pyaudio

# Option 2: Install Visual C++ Build Tools
# 1. Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 2. Download and install "Build Tools for Visual Studio"
# 3. Run: pip install pyaudio

# Option 3: Use pre-compiled wheel
# 1. Go to: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# 2. Download appropriate .whl file for your Python version
# 3. Install with: pip install [downloaded_file].whl
```

**Linux/macOS:**
```bash
# Ubuntu/Debian:
sudo apt-get install python3-pyaudio portaudio19-dev
pip install pyaudio

# macOS:
brew install portaudio
pip install pyaudio

# Alternative: Use conda
conda install pyaudio
```

#### 2. "Audio setup failed"
- Check if your microphone is connected and working
- Make sure no other application is using the microphone
- Try restarting your audio drivers

#### 3. "API Error: 401"
- Check your API key in the `.env` file
- Make sure your ElevenLabs account has sufficient credits

#### 4. "API Error: 404"
- Check your voice ID in the `.env` file
- Make sure the voice exists in your ElevenLabs account

#### 5. High latency
- Check your internet connection
- Try the Advanced version for lower latency
- Close other applications using audio
- Use wired internet connection

#### 6. No audio output
- Check your speaker volume
- Make sure the output device is working
- Try restarting the application

### Performance Tips

1. **Use wired internet** for lower latency
2. **Close other applications** using audio
3. **Use a good microphone** for better input quality
4. **Monitor CPU usage** - high CPU can cause audio dropouts
5. **Try the Advanced version** if you need lower latency
6. **Use a stable internet connection** - latency depends on network quality

## Supported Languages

ElevenLabs supports 29 languages for voice changing:
- English (USA, UK, Australia, Canada)
- Japanese, Chinese, German, Hindi
- French (France, Canada), Korean, Portuguese
- Italian, Spanish, Indonesian, Dutch, Turkish
- Filipino, Polish, Swedish, Bulgarian, Romanian
- Arabic (Saudi Arabia, UAE), Czech, Greek
- Finnish, Croatian, Malay, Slovak, Danish
- Tamil, Ukrainian, Russian

## API Usage and Costs

- **Billing**: Pay per second of audio processed
- **Rate Limits**: Depends on your ElevenLabs plan
- **Quality**: Professional-grade voice transformation
- **Latency**: 200-800ms with optimizations

## Contributing

Feel free to submit issues, feature requests, or pull requests!

## License

This project is open source. Please check ElevenLabs' terms of service for API usage. 