# STT→TTS Voice Changer Approach

## Overview

The STT→TTS (Speech-to-Text → Text-to-Speech) approach is a new voice changer implementation that provides better quality for non-English languages, especially Russian. This approach addresses the limitations of ElevenLabs' Speech-to-Speech API for certain languages.

## Why STT→TTS?

### Problems with Speech-to-Speech (STS) for Russian
- **Poor quality**: Russian speech often results in "chewing" or distorted output
- **Limited language support**: STS works best with English
- **Inconsistent results**: Quality varies significantly between languages

### Benefits of STT→TTS
- **Better quality**: Excellent results for all languages including Russian
- **Language flexibility**: Works with any language supported by Google STT
- **Consistent results**: Reliable quality across different languages
- **Text control**: Can modify or filter text before TTS conversion

## How It Works

```
Microphone → Google STT → Text Processing → ElevenLabs TTS → Speakers
     ↓           ↓              ↓              ↓            ↓
   Capture   → Transcribe   →  Clean Text   → Generate   →  Play
```

### 1. Audio Capture
- Records audio from microphone in 2-second chunks
- Uses 16kHz sample rate (optimal for Google STT)
- 16-bit PCM format

### 2. Speech-to-Text (Google Cloud)
- Sends audio chunks to Google Speech-to-Text API
- Configured for Russian language (`ru-RU`)
- Returns transcribed text with punctuation

### 3. Text Processing
- Cleans and formats the transcribed text
- Can add filters, translations, or modifications
- Prepares text for TTS conversion

### 4. Text-to-Speech (ElevenLabs)
- Sends text to ElevenLabs via WebSocket streaming
- Uses multilingual model for best quality
- Receives audio chunks in real-time

### 5. Audio Output
- Plays transformed audio through speakers
- Maintains real-time streaming
- Minimal latency between input and output

## Technical Details

### Audio Settings
- **Input Sample Rate**: 16kHz (Google STT requirement)
- **Output Sample Rate**: 44.1kHz (ElevenLabs standard)
- **Channels**: Mono (1 channel)
- **Format**: 16-bit PCM
- **Buffer Duration**: 2 seconds (optimal for STT accuracy)

### API Configuration
- **Google STT**: `latest_long` model for better accuracy
- **ElevenLabs TTS**: `eleven_multilingual_v2` model
- **WebSocket**: Streaming for low latency
- **Latency Optimization**: Level 4 (maximum)

### Performance Characteristics
- **Latency**: 500ms - 1.5s (slightly higher than STS)
- **Quality**: Excellent for all languages
- **Resource Usage**: Medium (requires Google Cloud)
- **Reliability**: High (separate services)

## Setup Requirements

### 1. ElevenLabs API Key
- Enable "Text to Speech" (Read/Write)
- Enable "Voices" (Read)
- Enable "Models" (Read)

### 2. Google Cloud Setup
- Create Google Cloud project
- Enable Speech-to-Text API
- Create service account with credentials
- Download JSON key file

### 3. Environment Variables
```bash
ELEVENLABS_API_KEY=your_elevenlabs_key
VOICE_ID=your_target_voice_id
GOOGLE_APPLICATION_CREDENTIALS=path/to/google_credentials.json
```

## Usage

### Quick Start
```bash
# Setup Google Cloud (first time only)
python setup_google_cloud.py

# Test the pipeline
python test_stt_tts_pipeline.py

# Run the voice changer
python realtime_stt_tts_voice_changer.py
```

### Via Launcher
```bash
python run_voice_changer.py
# Choose option 3: STT→TTS Version
```

## Comparison with STS Approach

| Feature | Speech-to-Speech | STT→TTS |
|---------|------------------|---------|
| **English Quality** | Excellent | Excellent |
| **Russian Quality** | Poor | Excellent |
| **Latency** | 200-800ms | 500ms-1.5s |
| **Setup Complexity** | Low | Medium |
| **API Costs** | ElevenLabs only | Google + ElevenLabs |
| **Language Support** | Limited | Extensive |
| **Text Control** | None | Full control |

## Cost Considerations

### Google Cloud Speech-to-Text
- **Free Tier**: 60 minutes/month
- **Paid**: $0.006 per 15 seconds
- **Typical Usage**: ~$0.24/hour of voice changing

### ElevenLabs Text-to-Speech
- **Free Tier**: 10,000 characters/month
- **Paid**: $0.30 per 1,000 characters
- **Typical Usage**: ~$0.18/hour of voice changing

### Total Cost
- **Combined**: ~$0.42/hour of voice changing
- **Monthly (1 hour/day)**: ~$12.60

## Troubleshooting

### Common Issues

#### 1. "Google Cloud credentials not found"
```bash
# Run the setup helper
python setup_google_cloud.py
```

#### 2. "No speech detected"
- Speak clearly and loudly
- Check microphone permissions
- Ensure quiet environment

#### 3. "High latency"
- Check internet connection
- Verify Google Cloud API is enabled
- Close other applications

#### 4. "Poor transcription quality"
- Use quiet environment
- Speak clearly and at normal pace
- Check microphone quality

### Performance Tips

1. **Environment**: Use quiet room with good microphone
2. **Speech**: Speak clearly and at normal pace
3. **Internet**: Use stable, high-speed connection
4. **Hardware**: Use quality microphone and speakers
5. **Settings**: Adjust buffer duration if needed

## Future Enhancements

### Potential Improvements
- **Text filtering**: Remove filler words, correct grammar
- **Translation**: Real-time language translation
- **Voice cloning**: Use custom cloned voices
- **Batch processing**: Process multiple chunks simultaneously
- **Local STT**: Use local speech recognition for privacy

### Customization Options
- **Language selection**: Support for 120+ languages
- **Voice selection**: Any ElevenLabs voice
- **Text processing**: Custom filters and modifications
- **Audio effects**: Add reverb, pitch shifting, etc.

## Conclusion

The STT→TTS approach provides a robust solution for multi-language voice changing, especially for Russian and other non-English languages. While it has slightly higher latency than the STS approach, it offers significantly better quality and reliability for international users.

For English speakers, the original STS approach may still be preferred due to lower latency. For Russian speakers or multi-language applications, the STT→TTS approach is highly recommended. 