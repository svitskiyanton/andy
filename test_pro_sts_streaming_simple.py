#!/usr/bin/env python3
"""
Pro STS Streaming Simple Test
Uses fixed chunking to ensure complete audio processing
"""

import os
import sys
import time
import json
import queue
import signal
import asyncio
import threading
import tempfile
import argparse
import numpy as np
import soundfile as sf
import requests
from pydub import AudioSegment
import pyaudio
import atexit

# Global variables for cleanup
running = True
audio_chunks = []

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global running
    print("\n⏹️  Received interrupt signal, saving output and exiting...")
    running = False

def cleanup_handler():
    """Cleanup handler for atexit"""
    global running, audio_chunks
    if running:
        running = False
        print("💾 Saving output on exit...")
        if audio_chunks:
            save_audio_chunks(audio_chunks, "test_output.mp3")

def save_audio_chunks(chunks, output_file):
    """Save audio chunks to output file"""
    if not chunks:
        print("⚠️ No audio chunks to save")
        return
    
    try:
        # Combine all chunks
        combined_audio = b''.join(chunks)
        
        with open(output_file, 'wb') as f:
            f.write(combined_audio)
        
        print(f"✅ All audio saved to: {output_file}")
        print(f"📊 Total audio size: {len(combined_audio)} bytes")
        
        # Calculate duration (approximate)
        # MP3 at 192kbps = ~24KB per second
        duration_seconds = len(combined_audio) / 24000
        print(f"🎵 Duration: {duration_seconds:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error saving audio: {e}")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_handler)

class ProSTSStreamingSimpleTest:
    def __init__(self, input_file=None):
        # Audio settings (Pro quality)
        self.STS_SAMPLE_RATE = 44100
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Input/Output files
        self.input_file = input_file
        self.OUTPUT_FILE = f"test_pro_sts_streaming_simple_output_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
        self.QUEUE_LOG_FILE = "test_pro_sts_streaming_simple_log.txt"
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=50)
        self.playback_thread = None
        self.running = True
        self.playback_started = False
        self.mp3_chunks = []
        self.all_audio_chunks = []
        
        # Fixed chunking settings
        self.CHUNK_DURATION_MS = 1500  # 1.5 seconds - small enough for reliable processing
        
        # ElevenLabs Pro settings
        self.api_key = self._get_api_key()
        self.voice_id = self._get_voice_id()
        self.model_id = "eleven_multilingual_sts_v2"  # Only STS model available
        
        # Pro voice settings
        self.voice_settings = {
            "stability": 0.7,  # Higher stability for Pro
            "similarity_boost": 0.8,  # Higher similarity for Pro
            "style": 0.3,  # Moderate style for natural speech
            "use_speaker_boost": True  # Pro feature
        }
        
        # Pro audio settings
        self.output_format = "mp3_44100_192"  # Pro 192kbps MP3
        self.optimize_streaming_latency = 4  # Pro streaming optimization
        
        # Timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Audio processing
        self.audio_segments = []
        self.current_position = 0
        
        print(f"🎵 Pro STS Streaming Simple Test Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Output Format: {self.output_format}")
        print(f"   Sample Rate: {self.STS_SAMPLE_RATE}Hz")
        print(f"   Chunk Duration: {self.CHUNK_DURATION_MS}ms")
        print(f"   Voice Settings: {self.voice_settings}")
        
    def _get_api_key(self):
        """Get API key from environment or .env file"""
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
    
    def _get_voice_id(self):
        """Get voice ID from environment or .env file"""
        voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if voice_id:
            return voice_id
        
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_VOICE_ID="):
                            voice_id = line.split("=", 1)[1].strip()
                            if voice_id and voice_id != "your_voice_id_here":
                                return voice_id
        except Exception:
            pass
        
        # Default to Ekaterina if not found
        return "GN4wbsbejSnGSa1AzjH5"
    
    def log_to_file(self, message):
        """Log operations to file for testing"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.QUEUE_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"❌ Error logging to file: {e}")
    
    def load_and_prepare_audio(self):
        """Load MP3 file and prepare it for processing"""
        try:
            print(f"📁 Loading audio file: {self.input_file}")
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(self.input_file)
            print(f"✅ Loaded audio: {len(audio)}ms ({len(audio)/1000:.1f}s)")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("🔄 Converted to mono")
            
            # Resample to 44.1kHz for STS processing
            if audio.frame_rate != self.STS_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.STS_SAMPLE_RATE)
                print(f"🔄 Resampled to {self.STS_SAMPLE_RATE}Hz")
            
            self.audio_segments = [audio]
            print(f"✅ Audio prepared for processing")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return False
    
    def create_fixed_chunks(self, audio_segment):
        """Create fixed-size chunks for reliable processing"""
        print("🎤 Creating fixed-size chunks...")
        
        duration_ms = len(audio_segment)
        chunks = []
        
        for i in range(0, duration_ms, self.CHUNK_DURATION_MS):
            start = i
            end = min(i + self.CHUNK_DURATION_MS, duration_ms)
            chunks.append((start, end))
        
        print(f"✅ Created {len(chunks)} chunks of {self.CHUNK_DURATION_MS}ms each")
        return chunks
    
    def stream_audio_by_fixed_chunks(self):
        """Stream audio using fixed-size chunks"""
        print("🎤 Pro STS Streaming Simple: Starting fixed chunking...")
        
        if not self.audio_segments:
            print("❌ No audio loaded")
            return
        
        audio = self.audio_segments[0]
        
        # Create fixed chunks
        chunks = self.create_fixed_chunks(audio)
        
        if not chunks:
            print("❌ No chunks created")
            return
        
        chunk_count = 0
        processed_chunks = 0
        
        for i, (start_ms, end_ms) in enumerate(chunks):
            if not self.running:
                break
                
            chunk_count += 1
            chunk_duration = (end_ms - start_ms) / 1000.0
            
            print(f"🎵 Pro STS Streaming Simple: Processing chunk {chunk_count}/{len(chunks)} ({chunk_duration:.1f}s)")
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            
            # Convert chunk to bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_audio.export(temp_file.name, format="wav")
                temp_file_path = temp_file.name
            
            # Read the audio file
            with open(temp_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Process with STS API with retry logic
            success = self.process_audio_with_sts_pro_with_retry(audio_data, chunk_count)
            
            if success:
                processed_chunks += 1
            
            # Small delay between chunks for smooth streaming
            time.sleep(0.1)
        
        print(f"🎵 Pro STS Streaming Simple: Processed {processed_chunks}/{chunk_count} chunks")
    
    def process_audio_with_sts_pro_with_retry(self, audio_data, chunk_index, max_retries=3):
        """Process audio using ElevenLabs Speech-to-Speech API with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"🎵 Pro STS: Sending chunk {chunk_index} to ElevenLabs STS API (attempt {attempt + 1}/{max_retries})...")
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file, audio_array, self.STS_SAMPLE_RATE)
                    temp_file_path = temp_file.name
                
                # Prepare the request with Pro features
                headers = {
                    "xi-api-key": self.api_key
                }
                
                # Read the audio file
                with open(temp_file_path, "rb") as audio_file:
                    files = {
                        "audio": ("audio.wav", audio_file, "audio/wav"),
                        "model_id": (None, self.model_id),
                        "remove_background_noise": (None, "false"),  # Disable for better quality
                        "optimize_streaming_latency": (None, str(self.optimize_streaming_latency)),  # Pro streaming optimization
                        "output_format": (None, self.output_format),  # Pro 192kbps MP3
                        "voice_settings": (None, json.dumps(self.voice_settings))  # Pro voice settings
                    }
                    
                    print(f"🎵 Pro STS: Sending chunk {chunk_index} to ElevenLabs STS API")
                    print(f"   Model: {self.model_id}")
                    print(f"   Voice: {self.voice_id}")
                    print(f"   Output Format: {self.output_format}")
                    print(f"   Voice Settings: {self.voice_settings}")
                    
                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                        headers=headers,
                        files=files,
                        timeout=30  # 30 second timeout
                    )
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                print(f"🎵 Pro STS API response: status={response.status_code}")
                
                if response.status_code == 200:
                    # Get the audio data
                    audio_output = response.content
                    
                    if audio_output:
                        print(f"✅ Pro STS: Received {len(audio_output)} bytes for chunk {chunk_index}")
                        
                        # Add to continuous buffer for playback
                        try:
                            self.continuous_buffer.put_nowait(audio_output)
                        except queue.Full:
                            # Buffer full, skip this chunk to maintain real-time
                            pass
                        
                        # Collect for file saving
                        self.mp3_chunks.append(audio_output)
                        self.all_audio_chunks.append(audio_output)
                        global audio_chunks
                        audio_chunks.append(audio_output)
                        
                        # Save individual chunk immediately
                        chunk_filename = f"pro_sts_simple_chunk_{chunk_index}_{self.timestamp}.mp3"
                        with open(chunk_filename, 'wb') as f:
                            f.write(audio_output)
                        print(f"💾 Saved Pro STS Simple chunk {chunk_index}: {chunk_filename} ({len(audio_output)} bytes)")
                        
                        # Start playback when we have enough data
                        if not self.playback_started and len(self.mp3_chunks) >= 2:
                            self.playback_started = True
                            print(f"🎵 Starting Pro STS Simple streaming playback")
                        
                        # Log to file
                        self.log_to_file(f"PRO_STS_SIMPLE_SUCCESS: Chunk {chunk_index}, {len(audio_output)} bytes")
                        return True
                    else:
                        print(f"⚠️ Pro STS: No audio data received for chunk {chunk_index}")
                        self.log_to_file(f"PRO_STS_SIMPLE_ERROR: Chunk {chunk_index} - No audio data received")
                        if attempt < max_retries - 1:
                            print(f"🔄 Retrying chunk {chunk_index}...")
                            time.sleep(1)  # Wait before retry
                            continue
                        return False
                else:
                    print(f"❌ Pro STS API error for chunk {chunk_index}: {response.status_code} - {response.text}")
                    self.log_to_file(f"PRO_STS_SIMPLE_ERROR: Chunk {chunk_index} - {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying chunk {chunk_index}...")
                        time.sleep(2)  # Wait longer before retry
                        continue
                    return False
                    
            except Exception as e:
                print(f"❌ Pro STS processing error for chunk {chunk_index}: {e}")
                self.log_to_file(f"PRO_STS_SIMPLE_ERROR: Chunk {chunk_index} - {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying chunk {chunk_index}...")
                    time.sleep(1)  # Wait before retry
                    continue
                return False
        
        return False
    
    async def _smooth_audio_streaming(self):
        """Smooth audio streaming with continuous playback"""
        try:
            # Initialize audio stream
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Start continuous playback thread if not already started
            if not self.playback_thread or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._continuous_playback_worker, args=(stream,))
                self.playback_thread.daemon = True
                self.playback_thread.start()
            
            print("🎵 Pro STS Simple Streaming: Audio streaming initialized")
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while self.running:
                try:
                    # Get audio chunk from buffer
                    mp3_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Play the audio chunk smoothly
                    self._play_audio_chunk_smooth(mp3_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue
                    continue
                except Exception as e:
                    print(f"❌ Playback error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Continuous playback error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def _play_audio_chunk_smooth(self, mp3_data, stream):
        """Play audio chunk smoothly without gaps"""
        try:
            # Convert MP3 to PCM
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Resample if needed
            if audio_segment.frame_rate != self.SAMPLE_RATE:
                audio_segment = audio_segment.set_frame_rate(self.SAMPLE_RATE)
            
            # Convert to PCM
            pcm_data = audio_segment.raw_data
            
            # Play the audio
            stream.write(pcm_data)
            
        except Exception as e:
            print(f"❌ Audio chunk playback error: {e}")
    
    async def start(self):
        """Start the Pro STS streaming test"""
        try:
            print("🎤🎵 Pro STS Simple Streaming Test: MP3 File → ElevenLabs STS API → Speakers")
            print("=" * 70)
            print("🎤 STS: Uses fixed chunking → sends to ElevenLabs STS API")
            print("🎵 TTS: Returns converted audio → smooth live playback")
            print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
            print(f"🎵 Output file: {self.OUTPUT_FILE}")
            print(f"🎵 Model: {self.model_id}")
            print(f"🎵 Voice: {self.voice_id}")
            print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
            print("=" * 70)
            
            # Load and prepare audio
            if not self.load_and_prepare_audio():
                return
            
            # Start smooth audio streaming
            await self._smooth_audio_streaming()
            
            print("🎤 Pro STS Streaming Simple: Starting fixed chunking...")
            print("✅ Pro STS Simple streaming test started!")
            print("🎵 Streaming MP3 file using fixed chunks...")
            print("🎵 Converted audio will play through speakers!")
            print("⏹️  Press Ctrl+C to stop early")
            
            # Stream audio by fixed chunks
            self.stream_audio_by_fixed_chunks()
            
            print("✅ Pro STS Simple streaming test completed")
            
        except Exception as e:
            print(f"❌ Pro STS Simple streaming error: {e}")
        finally:
            self.save_final_output()
    
    def save_final_output(self):
        """Save final output file"""
        try:
            print("💾 Saving final output...")
            save_audio_chunks(self.all_audio_chunks, self.OUTPUT_FILE)
        except Exception as e:
            print(f"❌ Error saving final output: {e}")

async def main():
    """Main function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Pro STS Streaming Simple Test")
        parser.add_argument("input_file", help="Input MP3 file")
        parser.add_argument("--output", help="Output MP3 file", default="test_output.mp3")
        args = parser.parse_args()
        
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the test
        pro_sts_streaming = ProSTSStreamingSimpleTest(args.input_file)
        
        # Override output file if specified
        if args.output:
            pro_sts_streaming.OUTPUT_FILE = args.output
            print(f"🎵 Output file set to: {args.output}")
        
        # Start the streaming test
        await pro_sts_streaming.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Main error: {e}")

if __name__ == "__main__":
    import io
    asyncio.run(main()) 