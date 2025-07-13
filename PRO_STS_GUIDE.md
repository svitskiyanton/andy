# Pro STS Test Script Guide

## Overview
This guide covers three Pro STS test scripts that use ElevenLabs Pro features for Speech-to-Speech voice conversion.

## Test Scripts

### 1. **Real-time Test** (`test_pro_sts.py`)
Captures a single short phrase from your microphone, sends it to ElevenLabs STS API with Pro features, and outputs the converted audio to your speakers.

### 2. **Batch Test** (`test_pro_sts_batch.py`)
Takes an MP3 file as input, sends it to ElevenLabs STS API in batch mode, and saves the converted audio to an output file.

### 3. **Streaming Test** (`test_pro_sts_streaming.py`) ⭐ **NEW**
Streams an MP3 file in real-time chunks to ElevenLabs STS API and plays the converted audio through speakers as it's processed.

## Pro Features Used

### 🎵 Audio Quality (Pro)
- **192kbps MP3 output format** (vs 128kbps for free tier)
- **44.1kHz sample rate** for enhanced quality
- **Optimized streaming latency** for real-time performance

### 🎭 Voice Settings (Pro)
- **Stability: 0.7** - Higher consistency across phrases
- **Similarity Boost: 0.8** - Enhanced voice cloning accuracy
- **Style: 0.3** - Natural speech patterns
- **Speaker Boost: True** - Pro-only feature for enhanced clarity

### ⚡ Performance (Pro)
- **Larger audio chunks** (4096 vs 1024) for better processing
- **Optimized streaming latency** (level 4) for reduced delays
- **Enhanced buffer management** for smoother playback

## Setup Instructions

### 1. Install Dependencies
```bash
pip install pyaudio numpy pydub soundfile python-dotenv requests websockets
```

### 2. Configure Environment
Copy `env_template.txt` to `.env` and fill in your credentials:

```bash
cp env_template.txt .env
```

Edit `.env` file:
```env
ELEVENLABS_API_KEY=your_actual_api_key_here
ELEVENLABS_VOICE_ID=your_preferred_voice_id_here
```

## Usage

### Real-time Test (Microphone Input)
```bash
python test_pro_sts.py
```

**How it works:**
1. 🎤 **Capture**: Records audio from microphone until silence detected
2. 📡 **Send**: Sends single chunk to ElevenLabs STS API with Pro settings
3. 🎵 **Receive**: Gets high-quality 192kbps MP3 response
4. 🔊 **Play**: Streams converted audio to speakers immediately

### Batch Test (MP3 File Input)
```bash
# Basic usage
python test_pro_sts_batch.py input_file.mp3

# With custom output file
python test_pro_sts_batch.py input_file.mp3 --output my_output.mp3
```

**How it works:**
1. 📁 **Load**: Loads MP3 file and splits into chunks (max 5 minutes each)
2. 📡 **Send**: Sends each chunk to ElevenLabs STS API with Pro settings
3. 🎵 **Receive**: Gets high-quality 192kbps MP3 responses
4. 💾 **Save**: Combines all chunks and saves to output file

### Streaming Test (Real-time MP3 Processing) ⭐ **NEW**
```bash
# Basic usage
python test_pro_sts_streaming.py input_file.mp3

# With custom output file
python test_pro_sts_streaming.py input_file.mp3 --output my_output.mp3
```

**How it works:**
1. 📁 **Load**: Loads MP3 file and prepares for streaming
2. 🔄 **Stream**: Processes file in 3-second chunks with 0.5s overlap
3. 📡 **Send**: Sends each chunk to ElevenLabs STS API with Pro settings
4. 🎵 **Receive**: Gets high-quality 192kbps MP3 responses
5. 🔊 **Play**: Streams converted audio to speakers in real-time
6. 💾 **Save**: Saves all chunks to output file

### Create Test Audio File
```bash
python create_test_audio.py
```
This creates a test MP3 file you can use for testing.

## Expected Output

### Real-time Test Output
```
🎵 Pro STS Test Configuration:
   Voice ID: GN4wbsbejSnGSa1AzjH5
   Model: eleven_multilingual_sts_v2
   Output Format: mp3_44100_192
   Sample Rate: 44100Hz
   Voice Settings: {'stability': 0.7, 'similarity_boost': 0.8, 'style': 0.3, 'use_speaker_boost': True}

🎤 Pro STS Test: Capturing single phrase...
🎤 Speak now (will stop after silence or max duration)...
✅ Audio input stream started

🎤 Pro STS: Processing 123456 bytes (3.2s) - silence detected
🎵 Pro STS: Sending audio to ElevenLabs STS API with Pro features...
🎵 Pro STS API response: status=200
✅ Pro STS: Received 45678 bytes of audio
💾 Saved Pro STS chunk: pro_sts_chunk_20241210_143022.mp3 (45678 bytes)
🎵 Starting Pro STS playback
```

### Batch Test Output
```
🎵 Pro STS Batch Test Configuration:
   Input File: test_audio_20241210_143022.mp3
   Output File: test_pro_sts_batch_output_20241210_143022.mp3
   Voice ID: GN4wbsbejSnGSa1AzjH5
   Model: eleven_multilingual_sts_v2
   Output Format: mp3_44100_192
   Sample Rate: 44100Hz
   Voice Settings: {'stability': 0.7, 'similarity_boost': 0.8, 'style': 0.3, 'use_speaker_boost': True}

📁 Loading audio file: test_audio_20241210_143022.mp3
✅ Loaded audio: 6000ms (6.0s)
📦 Single chunk processing
✅ Audio prepared: 1 chunks ready for processing

🎵 Pro STS Batch: Starting processing of 1 chunks...
🔄 Processing chunk 1/1
🎵 Pro STS: Processing chunk 1/1 (6000ms)
🎵 Pro STS: Sending chunk 1 to ElevenLabs STS API
🎵 Pro STS API response: status=200
✅ Pro STS: Received 45678 bytes for chunk 1
💾 Saved chunk 1: pro_sts_batch_chunk_1_20241210_143022.mp3 (45678 bytes)
✅ Chunk 1/1 processed successfully

🎵 Pro STS Batch: Processing complete!
   Total chunks: 1
   Successful: 1
   Failed: 0

💾 Combining 1 processed chunks...
✅ Combined audio saved to: test_pro_sts_batch_output_20241210_143022.mp3
📊 Total audio size: 45678 bytes
✅ Pro STS batch test completed successfully!
```

### Streaming Test Output
```
🎵 Pro STS Streaming Test Configuration:
   Input File: test_audio_20241210_143022.mp3
   Output File: test_pro_sts_streaming_output_20241210_143022.mp3
   Voice ID: GN4wbsbejSnGSa1AzjH5
   Model: eleven_multilingual_sts_v2
   Output Format: mp3_44100_192
   Sample Rate: 44100Hz
   Stream Chunk Duration: 3.0s
   Voice Settings: {'stability': 0.7, 'similarity_boost': 0.8, 'style': 0.3, 'use_speaker_boost': True}

📁 Loading audio file: test_audio_20241210_143022.mp3
✅ Loaded audio: 6000ms (6.0s)
🔄 Converted to mono
🔄 Resampled to 44100Hz
✅ Audio prepared for streaming

🎵 Pro STS Streaming: Audio streaming initialized
✅ Pro STS streaming test started!
🎵 Streaming MP3 file to ElevenLabs STS API...
🎵 Converted audio will play through speakers!

🎤 Pro STS Streaming: Starting audio stream from MP3 file...
🎵 Pro STS Streaming: Processing chunk 1 (3.0s)
🎵 Pro STS: Sending chunk 1 to ElevenLabs STS API with Pro features...
🎵 Pro STS: Sending chunk 1 to ElevenLabs STS API
🎵 Pro STS API response: status=200
✅ Pro STS: Received 45678 bytes for chunk 1
💾 Saved Pro STS streaming chunk 1: pro_sts_streaming_chunk_1_20241210_143022.mp3 (45678 bytes)
🎵 Starting Pro STS streaming playback

🎵 Pro STS Streaming: Processing chunk 2 (3.0s)
🎵 Pro STS: Sending chunk 2 to ElevenLabs STS API with Pro features...
🎵 Pro STS API response: status=200
✅ Pro STS: Received 45678 bytes for chunk 2
💾 Saved Pro STS streaming chunk 2: pro_sts_streaming_chunk_2_20241210_143022.mp3 (45678 bytes)

🎵 Pro STS Streaming: Reached end of audio file
🎵 Pro STS Streaming: Processed 2 chunks
🎵 Waiting for playback to complete...
✅ Pro STS streaming test completed
```

## Files Generated

### Real-time Test
- `test_pro_sts_output_YYYYMMDD_HHMMSS.mp3` - Combined audio output
- `pro_sts_chunk_YYYYMMDD_HHMMSS.mp3` - Individual audio chunk
- `test_pro_sts_log.txt` - Detailed operation log

### Batch Test
- `test_pro_sts_batch_output_YYYYMMDD_HHMMSS.mp3` - Combined audio output
- `pro_sts_batch_chunk_X_YYYYMMDD_HHMMSS.mp3` - Individual audio chunks
- `test_pro_sts_batch_log.txt` - Detailed operation log

### Streaming Test
- `test_pro_sts_streaming_output_YYYYMMDD_HHMMSS.mp3` - Combined audio output
- `pro_sts_streaming_chunk_X_YYYYMMDD_HHMMSS.mp3` - Individual audio chunks
- `test_pro_sts_streaming_log.txt` - Detailed operation log

## Troubleshooting

### No Audio Input (Real-time Test)
- Check microphone permissions
- Ensure microphone is not muted
- Try speaking louder or closer to microphone

### File Not Found (Batch/Streaming Tests)
- Verify the input MP3 file exists
- Check file path is correct
- Ensure file is a valid MP3 format

### API Errors
- Verify your ElevenLabs API key is correct
- Ensure you have Pro subscription for 192kbps features
   - Check internet connection

### Playback Issues
- Check speaker/headphone connections
- Ensure audio drivers are working
- Try different audio output device

## Pro vs Free Tier Differences

| Feature | Free Tier | Pro Tier |
|---------|-----------|----------|
| Audio Quality | 128kbps MP3 | **192kbps MP3** |
| Sample Rate | 22.05kHz | **44.1kHz** |
| Voice Settings | Basic | **Advanced (stability, similarity, style, speaker_boost)** |
| Streaming Latency | Basic | **Optimized (level 4)** |
| Processing Speed | Standard | **Enhanced** |
| Batch Processing | Limited | **Full support** |
| Real-time Streaming | Limited | **Full support** |

## Voice Selection Tips

- **Ekaterina (Russian)**: Good for Russian speech
- **George (British)**: Clear British accent
- **Rachel (American)**: Natural American accent
- **Domi (American)**: Professional voice
- **Bella (American)**: Warm, friendly tone

## Advanced Configuration

You can modify these settings in the scripts:

```python
# Pro voice settings
self.voice_settings = {
    "stability": 0.7,        # 0.0-1.0 (higher = more consistent)
    "similarity_boost": 0.8,  # 0.0-1.0 (higher = more similar to original)
    "style": 0.3,            # 0.0-1.0 (higher = more expressive)
    "use_speaker_boost": True # Pro feature for enhanced clarity
}

# Pro audio settings
self.output_format = "mp3_44100_192"  # Pro 192kbps MP3
self.optimize_streaming_latency = 4   # Pro streaming optimization

# Streaming settings (for streaming test)
self.STREAM_CHUNK_DURATION = 3.0  # 3 seconds per chunk
self.STREAM_OVERLAP = 0.5         # 0.5 seconds overlap
```

## Batch Processing Features

### Automatic Chunking
- Files longer than 5 minutes are automatically split
- Each chunk is processed separately
- Results are combined into final output

### Error Handling
- Individual chunk failures don't stop the entire process
- Failed chunks are logged with detailed error messages
- Successful chunks are saved even if some fail

### Progress Tracking
- Real-time progress updates for each chunk
- Detailed logging of all operations
- Summary statistics at completion

## Streaming Processing Features

### Real-time Chunking
- Processes MP3 file in 3-second chunks
- 0.5-second overlap between chunks for smooth transitions
- Real-time playback as chunks are processed

### Continuous Playback
- Starts playback after 3 chunks are buffered
- Smooth audio streaming without gaps
- Real-time conversion and playback

### Adaptive Processing
- Handles files of any length
- Automatic audio format conversion
- Optimized for real-time performance

## Performance Tips

### For Real-time Testing
- Use shorter phrases for faster response
- Ensure good microphone quality
- Minimize background noise

### For Batch Testing
- Use high-quality input files
- Process during off-peak hours for faster API response
- Monitor API usage limits

### For Streaming Testing
- Use files with clear speech
- Ensure stable internet connection
- Monitor system resources during streaming

## Support

### Getting Help
1. **Check logs**: Look at the generated log files
2. **Verify Pro status**: Ensure your ElevenLabs subscription is active
3. **Test with small files**: Start with short audio files
4. **Check audio devices**: Ensure microphone/speakers work

### Pro Support
- **ElevenLabs Pro Support**: Available for Pro subscribers
- **API Documentation**: https://elevenlabs.io/docs
- **Community**: ElevenLabs Discord server