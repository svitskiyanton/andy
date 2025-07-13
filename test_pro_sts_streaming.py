#!/usr/bin/env python3
"""
Test Script: Streaming STS with ElevenLabs Pro Features
Streams MP3 file input → ElevenLabs STS API → Speakers
Uses Pro subscription capabilities: 192kbps MP3, advanced voice settings, optimized streaming
"""

import asyncio
import json
import base64
import time
import numpy as np
import pyaudio
import threading
import queue
from pydub import AudioSegment
import io
import os
import tempfile
import soundfile as sf
from dotenv import load_dotenv
import requests
import websockets
import argparse
import sys
import signal
import atexit

# Load environment variables
load_dotenv()

# Global variable to hold the streaming test instance
global_streaming_test = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⏹️  Received interrupt signal, saving output and exiting...")
    if global_streaming_test:
        global_streaming_test.save_final_output()
    sys.exit(0)

def cleanup_handler():
    """Cleanup handler for atexit"""
    if global_streaming_test:
        print("💾 Saving output on exit...")
        global_streaming_test.save_final_output()

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_handler)

class ProSTSStreamingTest:
    def __init__(self, input_file=None):
        # Audio settings (Pro quality)
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.CHUNK_SIZE = 2048  # Larger chunk for Pro
        
        # Input/Output files
        self.input_file = input_file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_FILE = f"test_pro_sts_streaming_output_{timestamp}.mp3"
        
        # File logging
        self.QUEUE_LOG_FILE = "test_pro_sts_streaming_log.txt"
        
        # Control
        self.running = True
        
        # STS settings (Pro optimized)
        self.STS_SAMPLE_RATE = 44100  # Pro quality
        self.STS_CHANNELS = 1
        self.STS_FORMAT = pyaudio.paInt16
        self.STS_CHUNK_SIZE = 4096  # Larger for Pro
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=100)
        self.playback_thread = None
        
        # Playback control
        self.playback_started = False
        self.mp3_chunks = []
        self.all_audio_chunks = []
        
        # Streaming settings
        self.STREAM_CHUNK_DURATION = 3.0  # 3 seconds per chunk
        self.STREAM_CHUNK_SIZE = int(self.STS_SAMPLE_RATE * self.STREAM_CHUNK_DURATION)
        self.STREAM_OVERLAP = 0.5  # 0.5 seconds overlap between chunks
        self.MAX_CHUNK_DURATION = 10.0  # Maximum 10 seconds per chunk for API limits
        
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
        
        print(f"🎵 Pro STS Streaming Test Configuration:")
        print(f"   Input File: {self.input_file}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Output Format: {self.output_format}")
        print(f"   Sample Rate: {self.STS_SAMPLE_RATE}Hz")
        print(f"   Stream Chunk Duration: {self.STREAM_CHUNK_DURATION}s")
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
        """Load MP3 file and prepare it for streaming"""
        try:
            print(f"📁 Loading audio file: {self.input_file}")
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(self.input_file)
            print(f"✅ Loaded audio: {len(audio)}ms ({len(audio)/1000:.1f}s)")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                print("🔄 Converted to mono")
            
            # Resample to 44.1kHz if needed
            if audio.frame_rate != self.STS_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.STS_SAMPLE_RATE)
                print(f"🔄 Resampled to {self.STS_SAMPLE_RATE}Hz")
            
            self.audio_segments = [audio]
            print(f"✅ Audio prepared for streaming")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return False
    
    def get_next_audio_chunk(self):
        """Get next audio chunk for streaming"""
        if not self.audio_segments or self.current_position >= len(self.audio_segments[0]):
            return None
        
        # Calculate chunk boundaries with proper chunking
        start_pos = self.current_position
        chunk_duration_ms = min(self.STREAM_CHUNK_DURATION * 1000, self.MAX_CHUNK_DURATION * 1000)  # Convert to milliseconds
        end_pos = min(start_pos + chunk_duration_ms, len(self.audio_segments[0]))
        
        # Extract chunk
        chunk = self.audio_segments[0][start_pos:end_pos]
        
        # Update position (with overlap)
        overlap_ms = self.STREAM_OVERLAP * 1000  # Convert to milliseconds
        self.current_position = max(0, end_pos - overlap_ms)
        
        return chunk
    
    def stream_audio_chunks(self):
        """Stream audio chunks from MP3 file"""
        print("🎤 Pro STS Streaming: Starting audio stream from MP3 file...")
        
        chunk_count = 0
        
        while self.running:
            try:
                # Get next audio chunk
                audio_chunk = self.get_next_audio_chunk()
                
                if audio_chunk is None:
                    print("✅ Pro STS Streaming: Reached end of audio file")
                    break
                
                chunk_count += 1
                chunk_duration = len(audio_chunk) / 1000.0  # Convert to seconds
                
                print(f"🎵 Pro STS Streaming: Processing chunk {chunk_count} ({chunk_duration:.1f}s)")
                
                # Skip chunks that are too short (less than 0.5 seconds)
                if chunk_duration < 0.5:
                    print(f"⚠️ Skipping chunk {chunk_count} - too short ({chunk_duration:.1f}s)")
                    continue
                
                # Convert audio chunk to bytes
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    audio_chunk.export(temp_file.name, format="wav")
                    temp_file_path = temp_file.name
                
                # Read the audio file
                with open(temp_file_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                # Process with STS API
                success = self.process_audio_with_sts_pro(audio_data, chunk_count)
                
                if not success:
                    print(f"❌ Failed to process chunk {chunk_count}, continuing...")
                
                # Small delay between chunks for smooth streaming
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_count}: {e}")
                self.log_to_file(f"STREAMING_ERROR: Chunk {chunk_count} - {e}")
                continue
        
        print(f"🎵 Pro STS Streaming: Processed {chunk_count} chunks")
    
    def process_audio_with_sts_pro(self, audio_data, chunk_index):
        """Process audio using ElevenLabs Speech-to-Speech API with Pro features"""
        try:
            print(f"🎵 Pro STS: Sending chunk {chunk_index} to ElevenLabs STS API with Pro features...")
            
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
                    
                    # Save individual chunk immediately
                    chunk_filename = f"pro_sts_streaming_chunk_{chunk_index}_{self.timestamp}.mp3"
                    with open(chunk_filename, 'wb') as f:
                        f.write(audio_output)
                    print(f"💾 Saved Pro STS streaming chunk {chunk_index}: {chunk_filename} ({len(audio_output)} bytes)")
                    
                    # Start playback when we have enough data
                    if not self.playback_started and len(self.mp3_chunks) >= 3:
                        self.playback_started = True
                        print(f"🎵 Starting Pro STS streaming playback")
                    
                    # Log to file
                    self.log_to_file(f"PRO_STS_STREAMING_SUCCESS: Chunk {chunk_index}, {len(audio_output)} bytes")
                    return True
                else:
                    print(f"⚠️ Pro STS: No audio data received for chunk {chunk_index}")
                    self.log_to_file(f"PRO_STS_STREAMING_ERROR: Chunk {chunk_index} - No audio data received")
                    return False
            else:
                print(f"❌ Pro STS API error for chunk {chunk_index}: {response.status_code} - {response.text}")
                self.log_to_file(f"PRO_STS_STREAMING_ERROR: Chunk {chunk_index} - {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Pro STS processing error for chunk {chunk_index}: {e}")
            self.log_to_file(f"PRO_STS_STREAMING_ERROR: Chunk {chunk_index} - {e}")
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
            
            print("🎵 Pro STS Streaming: Audio streaming initialized")
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while self.running:
                try:
                    # Get audio chunk (blocking with timeout)
                    audio_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Decode and play immediately
                    self._play_audio_chunk_smooth(audio_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue waiting
                    continue
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    continue
            
            print("🎵 Pro STS streaming playback worker completed")
            
        except Exception as e:
            print(f"❌ Continuous playback worker error: {e}")
    
    def _play_audio_chunk_smooth(self, mp3_data, stream):
        """Play audio chunk with minimal processing for smooth playback"""
        try:
            if len(mp3_data) < 100:
                return
            
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            except Exception:
                return
            
            if len(audio_segment) == 0:
                return
            
            pcm_data = audio_segment.get_array_of_samples()
            pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
            
            # Play immediately without additional delays
            stream.write(pcm_float.astype(np.float32).tobytes())
            
        except Exception as e:
            # Silent error handling to avoid gaps
            pass
    
    async def start(self):
        """Start the Pro STS streaming test"""
        print("🎤🎵 Pro STS Streaming Test: MP3 File → ElevenLabs STS API → Speakers")
        print("=" * 70)
        print("🎤 STS: Streams MP3 file → sends chunks to ElevenLabs STS API")
        print("🎵 TTS: Returns converted audio → smooth live playback")
        print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
        print(f"🎵 Output file: {self.OUTPUT_FILE}")
        print(f"🎵 Model: {self.model_id}")
        print(f"🎵 Voice: {self.voice_id}")
        print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
        print("=" * 70)
        
        # Check if input file exists
        if not self.input_file or not os.path.exists(self.input_file):
            print(f"❌ Input file not found: {self.input_file}")
            print("   Please provide a valid MP3 file path")
            return
        
        # Clear log file
        if os.path.exists(self.QUEUE_LOG_FILE):
            os.remove(self.QUEUE_LOG_FILE)
        
        # Load and prepare audio
        if not self.load_and_prepare_audio():
            print("❌ Failed to load audio file")
            return
        
        # Start audio streaming
        await self._smooth_audio_streaming()
        
        # Start streaming thread
        streaming_thread = threading.Thread(target=self.stream_audio_chunks)
        streaming_thread.daemon = True
        streaming_thread.start()
        
        print("✅ Pro STS streaming test started!")
        print("🎵 Streaming MP3 file to ElevenLabs STS API...")
        print("🎵 Converted audio will play through speakers!")
        print("⏹️  Press Ctrl+C to stop early")
        
        try:
            # Keep main thread alive until streaming completes
            while self.running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Pro STS streaming test...")
            self.running = False
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            self.running = False
        finally:
            # Always save output, even if interrupted
            if self.all_audio_chunks:
                print("💾 Saving final output...")
                self.save_final_output()
                print("✅ Pro STS streaming test completed")
            else:
                print("⚠️ No audio chunks processed, nothing to save")
                print("✅ Pro STS streaming test completed")
    
    def save_final_output(self):
        """Save all TTS audio to a single MP3 file"""
        try:
            if self.all_audio_chunks:
                print(f"💾 Saving {len(self.all_audio_chunks)} audio chunks to {self.OUTPUT_FILE}...")
                
                # Combine all audio chunks
                combined_audio = b''.join(self.all_audio_chunks)
                
                # Save as MP3
                with open(self.OUTPUT_FILE, 'wb') as f:
                    f.write(combined_audio)
                
                print(f"✅ All audio saved to: {self.OUTPUT_FILE}")
                print(f"📊 Total audio size: {len(combined_audio)} bytes")
                print(f"🎵 Duration: {len(combined_audio) / (192 * 1024 / 8) * 1000:.1f} seconds")
                
                # Log final stats
                self.log_to_file(f"FINAL: Saved {len(self.all_audio_chunks)} chunks, {len(combined_audio)} bytes")
            else:
                print("⚠️ No audio chunks to save")
                self.log_to_file("FINAL: No audio chunks to save")
                
        except Exception as e:
            print(f"❌ Error saving final output: {e}")
            self.log_to_file(f"ERROR: Failed to save final output - {e}")

async def main():
    """Main function"""
    global global_streaming_test
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pro STS Streaming Test - Stream MP3 file with ElevenLabs STS API')
    parser.add_argument('input_file', help='Input MP3 file path')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found")
        print("   Please set your ElevenLabs API key in environment or .env file")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and start Pro STS streaming test
    pro_sts_streaming = ProSTSStreamingTest(args.input_file)
    global_streaming_test = pro_sts_streaming  # Assign to global variable
    
    # Override output file name if specified
    if args.output:
        pro_sts_streaming.OUTPUT_FILE = args.output
        print(f"🎵 Output file set to: {pro_sts_streaming.OUTPUT_FILE}")
    
    try:
        await pro_sts_streaming.start()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user, saving output...")
        pro_sts_streaming.save_final_output()
        print("✅ Output saved successfully")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("💾 Attempting to save any processed chunks...")
        pro_sts_streaming.save_final_output()
    finally:
        # Ensure output is saved even if no exception was caught
        if pro_sts_streaming.all_audio_chunks:
            print("💾 Final cleanup - saving output...")
            pro_sts_streaming.save_final_output()

if __name__ == "__main__":
    asyncio.run(main()) 