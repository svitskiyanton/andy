#!/usr/bin/env python3
"""
Test Script: Single Phrase STS with ElevenLabs Pro Features
Captures one short phrase from microphone → ElevenLabs STS API → Speakers
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

# Load environment variables
load_dotenv()

class ProSTSTest:
    def __init__(self):
        # Audio settings (Pro quality)
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.CHUNK_SIZE = 2048  # Larger chunk for Pro
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=50)
        self.playback_thread = None
        
        # Playback control
        self.playback_started = False
        self.mp3_chunks = []
        self.all_audio_chunks = []
        
        # Output file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_FILE = f"test_pro_sts_output_{timestamp}.mp3"
        
        # File logging
        self.QUEUE_LOG_FILE = "test_pro_sts_log.txt"
        
        # Control
        self.running = True
        
        # STS settings (Pro optimized)
        self.STS_SAMPLE_RATE = 44100  # Pro quality
        self.STS_CHANNELS = 1
        self.STS_FORMAT = pyaudio.paInt16
        self.STS_CHUNK_SIZE = 4096  # Larger for Pro
        
        # STS buffers
        self.audio_buffer = []
        self.silence_start = None
        self.max_silence_duration = 1.0
        
        # STS settings (Pro optimized)
        self.BUFFER_DURATION = 2.0  # Shorter for single phrase
        self.BUFFER_SIZE_STS = int(self.STS_SAMPLE_RATE * self.BUFFER_DURATION)
        self.SILENCE_DURATION = 0.8  # Shorter for single phrase
        self.SILENCE_THRESHOLD = 0.01
        
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
        
        print(f"🎵 Pro STS Test Configuration:")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Output Format: {self.output_format}")
        print(f"   Sample Rate: {self.STS_SAMPLE_RATE}Hz")
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
    
    def capture_single_phrase(self):
        """Capture a single phrase from microphone"""
        print("🎤 Pro STS Test: Capturing single phrase...")
        print("🎤 Speak now (will stop after silence or max duration)...")
        
        try:
            stream = self.audio.open(
                format=self.STS_FORMAT,
                channels=self.STS_CHANNELS,
                rate=self.STS_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.STS_CHUNK_SIZE
            )
            
            print("✅ Audio input stream started")
            
            # Buffer for accumulating audio until natural silence
            audio_buffer = b""
            silence_start = None
            silence_threshold = self.SILENCE_THRESHOLD
            min_phrase_duration = 0.8  # Minimum 0.8 seconds before processing
            max_phrase_duration = 10.0  # Maximum 10 seconds per phrase
            silence_duration = self.SILENCE_DURATION  # 0.8 seconds of silence to end phrase
            
            phrase_start_time = time.time()
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.STS_CHUNK_SIZE, exception_on_overflow=False)
                    audio_buffer += data
                    
                    # Check audio level for silence detection
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
                    
                    current_time = time.time()
                    phrase_duration = current_time - phrase_start_time
                    
                    # Check for silence
                    if audio_level < silence_threshold:
                        if silence_start is None:
                            silence_start = current_time
                    else:
                        silence_start = None
                    
                    # Determine when to process the phrase
                    should_process = False
                    
                    # Process if we have enough audio and silence detected
                    if (len(audio_buffer) > 0 and 
                        phrase_duration >= min_phrase_duration and
                        silence_start is not None and
                        (current_time - silence_start) >= silence_duration):
                        should_process = True
                    
                    # Also process if we've reached max duration (force processing)
                    elif phrase_duration >= max_phrase_duration and len(audio_buffer) > 0:
                        should_process = True
                    
                    if should_process:
                        print(f"🎤 Pro STS: Processing {len(audio_buffer)} bytes ({phrase_duration:.1f}s) - silence detected")
                        
                        # Send audio to ElevenLabs STS API
                        self.process_audio_with_sts_pro(audio_buffer)
                        
                        # Stop after processing one phrase
                        self.running = False
                        break
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    print(f"❌ Error in audio processing: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            print("✅ Audio capture completed")
            
        except Exception as e:
            print(f"❌ Audio capture error: {e}")
    
    def process_audio_with_sts_pro(self, audio_data):
        """Process audio using ElevenLabs Speech-to-Speech API with Pro features"""
        try:
            print("🎵 Pro STS: Sending audio to ElevenLabs STS API with Pro features...")
            
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
                
                print(f"🎵 Pro STS: Sending request to ElevenLabs STS API")
                print(f"   Model: {self.model_id}")
                print(f"   Voice: {self.voice_id}")
                print(f"   Output Format: {self.output_format}")
                print(f"   Voice Settings: {self.voice_settings}")
                
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                    headers=headers,
                    files=files
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            print(f"🎵 Pro STS API response: status={response.status_code}")
            
            if response.status_code == 200:
                # Get the audio data
                audio_output = response.content
                
                if audio_output:
                    print(f"✅ Pro STS: Received {len(audio_output)} bytes of audio")
                    
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
                    chunk_filename = f"pro_sts_chunk_{self.timestamp}.mp3"
                    with open(chunk_filename, 'wb') as f:
                        f.write(audio_output)
                    print(f"💾 Saved Pro STS chunk: {chunk_filename} ({len(audio_output)} bytes)")
                    
                    # Start playback immediately
                    if not self.playback_started:
                        self.playback_started = True
                        print(f"🎵 Starting Pro STS playback")
                    
                    # Log to file
                    self.log_to_file(f"PRO_STS_SUCCESS: {len(audio_output)} bytes")
                else:
                    print("⚠️ Pro STS: No audio data received")
                    self.log_to_file("PRO_STS_ERROR: No audio data received")
            else:
                print(f"❌ Pro STS API error: {response.status_code} - {response.text}")
                self.log_to_file(f"PRO_STS_ERROR: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Pro STS processing error: {e}")
            self.log_to_file(f"PRO_STS_ERROR: {e}")
    
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
            
            print("🎵 Pro STS: Audio streaming initialized")
            
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
            
            print("🎵 Pro STS playback worker completed")
            
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
        """Start the Pro STS test"""
        print("🎤🎵 Pro STS Test: Single Phrase Voice Changer")
        print("=" * 60)
        print("🎤 STS: Captures single phrase → sends to ElevenLabs STS API")
        print("🎵 TTS: Returns converted audio → smooth live playback")
        print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
        print(f"🎵 Output file: {self.OUTPUT_FILE}")
        print(f"🎵 Model: {self.model_id}")
        print(f"🎵 Voice: {self.voice_id}")
        print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
        print("=" * 60)
        
        # Clear log file
        if os.path.exists(self.QUEUE_LOG_FILE):
            os.remove(self.QUEUE_LOG_FILE)
        
        # Start audio streaming
        await self._smooth_audio_streaming()
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_single_phrase)
        capture_thread.daemon = True
        capture_thread.start()
        
        print("✅ Pro STS test started!")
        print("🎤 Speak a single phrase into your microphone...")
        print("🎵 Your voice will be converted and played back!")
        print("⏹️  Press Ctrl+C to stop early")
        
        try:
            # Keep main thread alive until capture completes
            while self.running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Pro STS test...")
            self.running = False
        
        # Wait for playback to complete
        if self.playback_thread and self.playback_thread.is_alive():
            print("🎵 Waiting for playback to complete...")
            self.playback_thread.join(timeout=5)
        
        self.save_final_output()
        print("✅ Pro STS test completed")
    
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
    # Check prerequisites
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found")
        print("   Please set your ElevenLabs API key in environment or .env file")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and start Pro STS test
    pro_sts_test = ProSTSTest()
    await pro_sts_test.start()

if __name__ == "__main__":
    asyncio.run(main()) 