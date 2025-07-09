# Pro Speech-to-Speech Voice Changer Guide

## Overview

This Pro-optimized Speech-to-Speech (STS) voice changer leverages your ElevenLabs Pro subscription to provide real-time voice transformation with minimal latency and maximum quality.

## Pro Features Leveraged

### 🚀 Performance Features
- **10 Concurrent Requests**: Pro allows up to 10 simultaneous STS requests
- **192 kbps Audio Quality**: Pro supports high-quality audio output
- **44.1kHz PCM Output**: Professional-grade audio format
- **Turbo Models**: Faster processing with `eleven_turbo_v2`
- **Priority Processing**: Pro requests get priority in the queue
- **100ms Latency Target**: Optimized for real-time performance

### 🎯 Voice Features
- **Voice Cloning**: Create and use your own voice clone
- **Multiple Voice Support**: Switch between different voices
- **Real-time Streaming**: Continuous audio processing
- **Silence Detection**: Smart phrase detection and processing

## Setup

### 1. Environment Variables
```bash
export ELEVENLABS_API_KEY="your_pro_api_key_here"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Test Pro Capabilities
```bash
python test_pro_sts.py
```

This will verify:
- ✅ Pro subscription status
- ✅ Available Turbo models
- ✅ Voice cloning capabilities
- ✅ STS endpoint availability
- ✅ Audio device compatibility

## Usage

### Basic Usage
```bash
python pro_speech_to_speech.py
```

### Configuration Options

Edit `pro_speech_to_speech.py` to customize:

```python
# Voice settings
self.VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Change voice
self.MODEL_ID = "eleven_turbo_v2"  # Use Turbo model

# Performance settings
self.LATENCY_TARGET = 0.1  # 100ms target
self.MAX_CONCURRENT_REQUESTS = 10  # Pro limit
self.AUDIO_QUALITY = 192  # Pro quality

# Audio settings
self.SAMPLE_RATE = 44100  # Pro supports 44.1kHz
self.SILENCE_THRESHOLD = 0.01  # Adjust sensitivity
self.MIN_PHRASE_DURATION = 0.5  # Minimum phrase length
self.MAX_PHRASE_DURATION = 10.0  # Maximum phrase length
```

## Voice Cloning (Pro Feature)

### Create Your Voice Clone

1. **Via ElevenLabs Dashboard**:
   - Go to https://elevenlabs.io/voice-cloning
   - Upload audio samples of your voice
   - Name your clone (e.g., "My Voice Clone")

2. **Update Configuration**:
   ```python
   self.ENABLE_VOICE_CLONING = True
   self.CLONE_VOICE_NAME = "My Voice Clone"
   ```

3. **The script will automatically**:
   - Detect your existing voice clone
   - Use it for STS processing
   - Fall back to default voice if not found

## Performance Optimization

### Latency Optimization
- **Target**: 100ms end-to-end latency
- **Optimization**: Uses Turbo models and priority processing
- **Monitoring**: Latency is logged for performance tracking

### Throughput Optimization
- **Concurrent Requests**: Up to 10 simultaneous STS requests
- **Queue Management**: Smart buffering prevents overflow
- **Memory Management**: Efficient audio buffer management

### Audio Quality
- **Input**: 44.1kHz, 16-bit PCM
- **Output**: 44.1kHz, 192 kbps MP3
- **Processing**: Real-time streaming with silence detection

## Troubleshooting

### Common Issues

1. **"ELEVENLABS_API_KEY environment variable is required"**
   ```bash
   export ELEVENLABS_API_KEY="your_key_here"
   ```

2. **Audio device not found**
   - Run `python test_pro_sts.py` to check devices
   - Ensure microphone is connected and working

3. **High latency**
   - Check internet connection
   - Reduce `LATENCY_TARGET` if needed
   - Monitor `pro_sts_voice_changer.log`

4. **Pro features not working**
   - Verify Pro subscription status
   - Check API key permissions
   - Run capability test: `python test_pro_sts.py`

### Log Files
- **Main log**: `pro_sts_voice_changer.log`
- **Performance metrics**: Latency and throughput tracking
- **Error details**: Detailed error logging

## Advanced Configuration

### Custom Voice Settings
```python
# Use a specific voice
self.VOICE_ID = "your_voice_id_here"

# Use different model
self.MODEL_ID = "eleven_multilingual_v2"  # For multiple languages
```

### Audio Processing
```python
# Adjust silence detection
self.SILENCE_THRESHOLD = 0.005  # More sensitive
self.SILENCE_THRESHOLD = 0.02   # Less sensitive

# Adjust phrase timing
self.MIN_PHRASE_DURATION = 0.3  # Shorter phrases
self.MAX_PHRASE_DURATION = 15.0  # Longer phrases
```

### Performance Tuning
```python
# Adjust buffer sizes
self.BUFFER_SIZE = 4096  # Smaller for lower latency
self.BUFFER_SIZE = 16384  # Larger for stability

# Adjust chunk sizes
self.CHUNK_SIZE = 512   # Smaller chunks
self.CHUNK_SIZE = 2048  # Larger chunks
```

## Pro Subscription Benefits

### What You Get
- ✅ **10 concurrent requests** (vs 2 for free)
- ✅ **192 kbps audio quality** (vs 128 kbps)
- ✅ **44.1kHz PCM output** (vs 22.05kHz)
- ✅ **Turbo models** (faster processing)
- ✅ **Priority processing** (faster queue)
- ✅ **Voice cloning** (create custom voices)
- ✅ **Higher character limits** (more usage)

### Usage Monitoring
The script logs your usage and can help you monitor:
- Character count used
- Request frequency
- Latency performance
- Error rates

## Real-time Performance

### Expected Performance
- **Latency**: 100-200ms end-to-end
- **Quality**: 192 kbps, 44.1kHz
- **Concurrency**: Up to 10 simultaneous requests
- **Reliability**: Automatic error recovery

### Monitoring
Watch the console output for:
- Real-time latency measurements
- Audio processing statistics
- Error messages and warnings
- Performance metrics

## Support

### Getting Help
1. **Check logs**: `pro_sts_voice_changer.log`
2. **Run tests**: `python test_pro_sts.py`
3. **Verify Pro status**: Check subscription in ElevenLabs dashboard
4. **Check audio**: Ensure microphone and speakers work

### Pro Support
- **ElevenLabs Pro Support**: Available for Pro subscribers
- **API Documentation**: https://elevenlabs.io/docs
- **Community**: ElevenLabs Discord server

---

**Enjoy your Pro Speech-to-Speech voice changer!** 🎤✨ 