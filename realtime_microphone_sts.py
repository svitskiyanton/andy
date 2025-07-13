#!/usr/bin/env python3
"""
Real-time Microphone STS Voice Changer
=====================================

Processes live microphone input using VAD chunking and ElevenLabs STS API.
Saves input and output chunks, streams playback in real-time.

Features:
- Real-time microphone input
- VAD-based natural phrase chunking
- ElevenLabs STS API processing
- Live audio streaming playback
- Input/output chunk saving
- Pro voice settings
"""

import os
import sys
import time
import signal
import atexit
import tempfile
import threading
import queue
import io
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import soundfile as sf
import webrtcvad
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

# Configuration
VOICE_ID = "GN4wbsbejSnGSa1AzjH5"  # Pro voice
MODEL = "eleven_multilingual_sts_v2"
OUTPUT_FORMAT = "mp3_44100_192"

# Voice settings for Pro
VOICE_SETTINGS = {
    "stability": 0.8,
    "similarity_boost": 0.85,
    "style": 0.2,
    "use_speaker_boost": True
}

# Audio settings - UPDATED to match working script quality
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048  # Larger chunks like working script (was 480)
VAD_MODE = 3  # Aggressive mode
SILENCE_DURATION = 0.5  # Seconds of silence to end phrase
MIN_PHRASE_DURATION = 1.0  # Minimum phrase duration (was 0.3)
MAX_PHRASE_DURATION = 8.0  # Maximum phrase duration (was 10.0)
SILENCE_THRESHOLD = 0.008  # Audio level threshold (like working script)

# Output settings
SAVE_CHUNKS = True
CHUNKS_DIR = "microphone_chunks"
OUTPUT_FILE = f"microphone_sts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

class MicrophoneSTSProcessor:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_running = False
        self.playback_queue = queue.Queue()
        
        # Chunk tracking
        self.input_chunks = []
        self.output_chunks = []
        self.combined_output = AudioSegment.empty()
        
        # Create chunks directory
        if SAVE_CHUNKS:
            Path(CHUNKS_DIR).mkdir(exist_ok=True)
        
        # Setup cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Received interrupt signal, stopping...")
        self.stop()
        self.save_final_output()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("🧹 Cleanup completed")
    
    def start_microphone_stream(self):
        """Start microphone input stream with direct reading (like working script)"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print("🎤 Microphone stream started (high-quality direct reading)")
            print(f"   Sample Rate: {SAMPLE_RATE}Hz")
            print(f"   Chunk Size: {CHUNK_SIZE} bytes")
            print(f"   Format: 16-bit PCM")
            return True
        except Exception as e:
            print(f"❌ Failed to start microphone: {e}")
            return False
    
    def collect_phrase_direct(self):
        """Collect a complete phrase using direct stream reading (like working script)"""
        print("🎤 Listening for speech...")
        
        audio_buffer = b""
        silence_start = None
        phrase_start_time = time.time()
        
        while self.is_running:
            try:
                # Read audio directly from stream (like working script)
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_buffer += data
                
                # Calculate audio level (like working script)
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
                
                current_time = time.time()
                phrase_duration = current_time - phrase_start_time
                
                # Detect silence using audio level
                if audio_level < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = current_time
                else:
                    silence_start = None
                
                # Check if phrase should end
                should_end = False
                
                # End if silence detected for required duration
                if (len(audio_buffer) > 0 and 
                    phrase_duration >= MIN_PHRASE_DURATION and
                    silence_start is not None and
                    (current_time - silence_start) >= SILENCE_DURATION):
                    should_end = True
                    print(f"🎤 Phrase end: silence detected for {current_time - silence_start:.1f}s")
                
                # End if maximum duration reached
                elif phrase_duration >= MAX_PHRASE_DURATION and len(audio_buffer) > 0:
                    should_end = True
                    print(f"⚠️  Max duration reached ({MAX_PHRASE_DURATION}s), ending phrase")
                
                if should_end:
                    break
                    
            except Exception as e:
                print(f"❌ Error reading audio: {e}")
                break
        
        if len(audio_buffer) == 0:
            return None
        
        # Calculate duration
        duration = len(audio_buffer) / (SAMPLE_RATE * 2)  # 16-bit = 2 bytes
        
        if duration < MIN_PHRASE_DURATION:
            print(f"⚠️  Phrase too short ({duration:.2f}s < {MIN_PHRASE_DURATION}s), skipping")
            return None
        
        # Convert to AudioSegment
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        
        print(f"✅ Collected phrase: {duration:.2f}s")
        return audio_segment
    
    def save_input_chunk(self, chunk, chunk_num):
        """Save input chunk to file"""
        if not SAVE_CHUNKS:
            return
        
        filename = f"{CHUNKS_DIR}/input_chunk_{chunk_num:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        chunk.export(filename, format="wav")
        print(f"💾 Saved input chunk: {filename}")
        return filename
    
    def save_output_chunk(self, audio_data, chunk_num):
        """Save output chunk to file"""
        if not SAVE_CHUNKS:
            return
        
        filename = f"{CHUNKS_DIR}/output_chunk_{chunk_num:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        
        # Convert bytes to AudioSegment
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment.export(filename, format="mp3")
        print(f"💾 Saved output chunk: {filename}")
        return filename
    
    def process_chunk_with_sts(self, audio_chunk, chunk_num):
        """Process audio chunk with ElevenLabs STS API"""
        print(f"🎵 STS: Processing chunk {chunk_num}")
        
        # Save input chunk
        input_file = self.save_input_chunk(audio_chunk, chunk_num)
        
        # Convert to WAV for API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_chunk.export(temp_file.name, format="wav")
            temp_wav_path = temp_file.name
        
        try:
            # Prepare API request - SAME AS sts_streaming_pipeline_debug.py
            headers = {
                "xi-api-key": self.api_key
            }
            
            with open(temp_wav_path, "rb") as audio_file:
                files = {
                    "audio": ("audio.wav", audio_file, "audio/wav"),
                    "model_id": (None, MODEL),
                    "remove_background_noise": (None, "false"),
                    "optimize_streaming_latency": (None, "false"),
                    "output_format": (None, OUTPUT_FORMAT),
                    "voice_settings": (None, json.dumps(VOICE_SETTINGS))
                }
                
                print(f"🎵 STS: Sending chunk {chunk_num} to ElevenLabs STS API")
                print(f"   Model: {MODEL}")
                print(f"   Voice: {VOICE_ID}")
                print(f"   Output Format: {OUTPUT_FORMAT}")
                
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/speech-to-speech/{VOICE_ID}/stream",
                    headers=headers,
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    audio_data = response.content
                    print(f"✅ STS: Received {len(audio_data)} bytes for chunk {chunk_num}")
                    
                    # Save output chunk
                    output_file = self.save_output_chunk(audio_data, chunk_num)
                    
                    # Convert to AudioSegment for playback
                    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                    
                    return audio_segment, output_file
                else:
                    print(f"❌ STS API error: {response.status_code} - {response.text}")
                    return None, None
                    
        except Exception as e:
            print(f"❌ STS processing error: {e}")
            return None, None
        finally:
            # Cleanup temp file
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
    
    def play_audio_chunk(self, audio_segment):
        """Play audio chunk"""
        try:
            print(f"🔊 Playing audio chunk ({len(audio_segment.raw_data)} bytes)")
            play(audio_segment)
            print("✅ Audio chunk played successfully")
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    def process_live_audio(self):
        """Main processing loop"""
        chunk_num = 0
        
        print("🎤🎵 Real-time Microphone STS Voice Changer (UPDATED)")
        print("=" * 60)
        print("🎤 Audio Level: Detects natural phrase boundaries")
        print("🎵 STS: Sends chunks to ElevenLabs STS API")
        print("🔊 Playback: Smooth real-time audio streaming")
        print(f"📁 Output file: {OUTPUT_FILE}")
        print(f"🎵 Model: {MODEL}")
        print(f"🎵 Voice: {VOICE_ID}")
        print("=" * 60)
        print("🎵 Streaming: Microphone → Audio Level Chunks → STS → Playback")
        print("🎵 Converted audio will play through speakers!")
        print("⏹️  Press Ctrl+C to stop")
        print()
        print("🔧 UPDATED: High-quality audio capture (44.1kHz, 2048-byte chunks)")
        print("🔧 UPDATED: Audio level-based silence detection (threshold: 0.008)")
        print("🔧 UPDATED: Better phrase duration limits (1.0s - 8.0s)")
        print()
        
        while self.is_running:
            # Collect phrase using direct audio reading
            audio_chunk = self.collect_phrase_direct()
            
            if audio_chunk is None:
                continue
            
            chunk_num += 1
            print(f"\n🎵 Processing chunk {chunk_num}")
            
            # Process with STS API
            processed_chunk, output_file = self.process_chunk_with_sts(audio_chunk, chunk_num)
            
            if processed_chunk is not None:
                # Add to combined output
                self.combined_output += processed_chunk
                
                # Play the processed audio
                self.play_audio_chunk(processed_chunk)
                
                print(f"✅ Chunk {chunk_num} processed successfully")
            else:
                print(f"❌ Failed to process chunk {chunk_num}")
            
            print("-" * 40)
    
    def save_final_output(self):
        """Save final combined output"""
        if len(self.combined_output) > 0:
            print(f"\n💾 Saving final output...")
            self.combined_output.export(OUTPUT_FILE, format="mp3")
            print(f"✅ All audio saved to: {OUTPUT_FILE}")
            print(f"📊 Total audio size: {len(self.combined_output.raw_data)} bytes")
        else:
            print("⚠️  No audio to save")
    
    def _get_api_key(self):
        """Get API key from environment or .env file - SAME AS sts_streaming_pipeline_debug.py"""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            return api_key
        
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            if api_key and api_key != "your_api_key_here":
                                return api_key
        except Exception:
            pass
        
        print("🔑 ElevenLabs API Key not found.")
        print("Please enter your ElevenLabs API key:")
        api_key = input("API Key: ").strip()
        
        if api_key:
            try:
                with open(".env", "w") as f:
                    f.write(f"ELEVENLABS_API_KEY={api_key}\n")
                print("✅ API key saved to .env file for future use")
            except Exception:
                print("⚠️ Could not save API key to file")
            
            return api_key
        
        return None

    def start(self):
        """Start the microphone STS processor"""
        self.api_key = self._get_api_key()
        if not self.api_key:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        print("✅ Prerequisites check passed")
        print(f"🎵 Real-time Microphone STS Configuration:")
        print(f"   Voice ID: {VOICE_ID}")
        print(f"   Model: {MODEL}")
        print(f"   Voice Settings: {VOICE_SETTINGS}")
        print(f"   Output Format: {OUTPUT_FORMAT}")
        print(f"   Save Chunks: {SAVE_CHUNKS}")
        if SAVE_CHUNKS:
            print(f"   Chunks Directory: {CHUNKS_DIR}")
        print(f"   Output File: {OUTPUT_FILE}")
        print()
        
        # Start microphone stream
        if not self.start_microphone_stream():
            return False
        
        self.is_running = True
        
        try:
            self.process_live_audio()
        except KeyboardInterrupt:
            print("\n🛑 Stopping microphone STS processor...")
        finally:
            self.stop()
            self.save_final_output()
    
    def stop(self):
        """Stop the processor"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

def main():
    """Main function"""
    processor = MicrophoneSTSProcessor()
    processor.start()

if __name__ == "__main__":
    main() 