#!/usr/bin/env python3
"""
Ultra-Low Latency Voice Changer using ElevenLabs Speech-to-Speech API
Maximum optimizations for minimal latency and consistent performance
"""

import os
import sys
import time
import threading
import queue
import signal
import collections
from typing import Optional, List
from dotenv import load_dotenv
import elevenlabs
import pyaudio
import requests
import wave
import io
import numpy as np

class UltraLowLatencyVoiceChanger:
    def __init__(self, api_key: str, voice_id: str):
        """
        Initialize the ultra-low latency voice changer
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Target voice ID for transformation
        """
        self.api_key = api_key
        self.voice_id = voice_id
        
        # Set API key for ElevenLabs
        elevenlabs.set_api_key(api_key)
        
        # Audio processing queues - minimal sizes for low latency
        self.input_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Control flags
        self.running = True
        
        # Audio settings - ultra-low latency optimized
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 1024  # Very small chunks (64ms)
        self.buffer_duration = 0.5  # Only 0.5 seconds for minimal latency
        
        # Audio objects
        self.pyaudio = None
        self.input_stream = None
        self.output_stream = None
        
        # Statistics
        self.processed_chunks = 0
        self.start_time = None
        self.latency_history = collections.deque(maxlen=20)
        
        # Audio buffer for batching
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Performance tracking
        self.last_process_time = 0
        self.min_interval = 0.3  # Very short minimum interval
        self.silence_threshold = 50  # Lower threshold for more sensitive detection
        
        # Pre-allocated buffers for efficiency
        self.wav_buffer = io.BytesIO()
        
    def setup_audio(self):
        """Setup audio input and output streams"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Get default devices
            input_device = self.pyaudio.get_default_input_device_info()
            output_device = self.pyaudio.get_default_output_device_info()
            
            print(f"🎤 Input Device: {input_device['name']}")
            print(f"🔊 Output Device: {output_device['name']}")
            print(f"📊 Sample Rate: {self.rate}Hz, Channels: {self.channels}")
            print(f"📦 Chunk Size: {self.chunk_size} samples ({self.chunk_size/self.rate*1000:.0f}ms)")
            print(f"🔄 Buffer Duration: {self.buffer_duration}s")
            print(f"⚡ Ultra-Low Latency Mode")
            
            # Setup input stream with minimal buffer
            self.input_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_host_api_specific_stream_info=None
            )
            
            # Setup output stream with minimal buffer
            self.output_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_host_api_specific_stream_info=None
            )
            
            print("✅ Audio setup complete")
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            sys.exit(1)
    
    def capture_audio(self):
        """Capture audio from microphone with ultra-low latency"""
        print("🎤 Starting audio capture...")
        
        try:
            while self.running:
                try:
                    # Read audio data with non-blocking
                    audio_data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Check if audio has actual content (not just silence)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    if np.abs(audio_array).mean() > self.silence_threshold:
                        # Add to buffer
                        with self.buffer_lock:
                            self.audio_buffer.append(audio_data)
                            
                            # Check if we have enough audio for processing
                            buffer_samples = len(self.audio_buffer) * self.chunk_size
                            buffer_duration = buffer_samples / self.rate
                            
                            if buffer_duration >= self.buffer_duration:
                                # Combine all chunks into one buffer
                                combined_audio = b''.join(self.audio_buffer)
                                self.input_queue.put(combined_audio)
                                self.audio_buffer = []  # Clear buffer
                    else:
                        # For silence, minimal delay
                        time.sleep(0.005)
                            
                except Exception as e:
                    print(f"⚠️ Audio capture error: {e}")
                    time.sleep(0.005)
                    
        except KeyboardInterrupt:
            print("\n🛑 Audio capture interrupted")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
    
    def pcm_to_wav_bytes_fast(self, pcm_data, sample_rate=16000, channels=1, sample_width=2):
        """Fast PCM to WAV conversion using pre-allocated buffer"""
        self.wav_buffer.seek(0)
        self.wav_buffer.truncate()
        
        with wave.open(self.wav_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        
        return self.wav_buffer.getvalue()
    
    def process_audio(self):
        """Process audio through ElevenLabs Speech-to-Speech API with ultra-low latency"""
        print("🔄 Starting voice transformation...")
        
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream"
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Ultra-optimized parameters for minimum latency
        params = {
            "optimize_streaming_latency": 4,  # Maximum optimizations
            "output_format": "pcm_16000",
            "model_id": "eleven_multilingual_sts_v2",
            "voice_settings": {
                "stability": 0.2,  # Very low stability for fastest processing
                "similarity_boost": 0.6
            }
        }
        
        while self.running:
            try:
                # Rate limiting to prevent API overload
                current_time = time.time()
                if current_time - self.last_process_time < self.min_interval:
                    time.sleep(0.005)  # Very short sleep
                    continue
                
                if not self.input_queue.empty():
                    start_time = time.time()
                    self.last_process_time = start_time
                    
                    # Get audio buffer
                    audio_buffer = self.input_queue.get(timeout=0.05)  # Shorter timeout
                    
                    # Convert to WAV using fast method
                    wav_data = self.pcm_to_wav_bytes_fast(audio_buffer)
                    
                    # Prepare files for upload
                    files = {
                        'audio': ('audio.wav', wav_data, 'audio/wav')
                    }
                    
                    # Send to ElevenLabs API with minimal timeout
                    response = requests.post(
                        url,
                        headers=headers,
                        params=params,
                        files=files,
                        stream=True,
                        timeout=5  # Very short timeout
                    )
                    
                    if response.status_code == 200:
                        # Collect all audio data efficiently
                        audio_chunks = []
                        for chunk in response.iter_content(chunk_size=8192):  # Larger chunks
                            if chunk:
                                audio_chunks.append(chunk)
                        
                        if audio_chunks:
                            # Combine all chunks
                            combined_audio = b''.join(audio_chunks)
                            
                            # Split into smaller chunks for playback
                            chunk_size_bytes = self.chunk_size * 2  # 16-bit = 2 bytes per sample
                            for i in range(0, len(combined_audio), chunk_size_bytes):
                                chunk = combined_audio[i:i + chunk_size_bytes]
                                if len(chunk) == chunk_size_bytes:  # Only full chunks
                                    self.output_queue.put(chunk)
                            
                            # Calculate latency
                            end_time = time.time()
                            latency = (end_time - start_time) * 1000
                            self.latency_history.append(latency)
                            self.processed_chunks += 1
                            
                    else:
                        print(f"❌ API Error: {response.status_code} - {response.text}")
                        
            except queue.Empty:
                continue
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error: {e}")
                time.sleep(0.05)  # Short delay
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                time.sleep(0.05)  # Short delay
    
    def play_audio(self):
        """Play transformed audio through speakers with minimal latency"""
        print("🔊 Starting audio output...")
        
        try:
            while self.running:
                try:
                    if not self.output_queue.empty():
                        audio_data = self.output_queue.get(timeout=0.05)  # Shorter timeout
                        self.output_stream.write(audio_data)
                    else:
                        time.sleep(0.0005)  # Very short sleep
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n🛑 Audio output interrupted")
        except Exception as e:
            print(f"❌ Audio output error: {e}")
        finally:
            if self.output_stream:
                self.output_stream.stop_stream()
    
    def monitor_performance(self):
        """Monitor and display performance statistics"""
        self.start_time = time.time()
        
        while self.running:
            try:
                time.sleep(1)  # Update every second for more frequent feedback
                
                if self.start_time and self.latency_history:
                    elapsed = time.time() - self.start_time
                    chunks_per_second = self.processed_chunks / elapsed if elapsed > 0 else 0
                    avg_latency = sum(self.latency_history) / len(self.latency_history)
                    min_latency = min(self.latency_history) if self.latency_history else 0
                    max_latency = max(self.latency_history) if self.latency_history else 0
                    
                    print(f"📊 Stats: {self.processed_chunks} chunks processed, "
                          f"{chunks_per_second:.1f} chunks/sec, "
                          f"Latency: {avg_latency:.0f}ms (min: {min_latency:.0f}ms, max: {max_latency:.0f}ms)")
                    
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Cleanup audio resources"""
        print("🧹 Cleaning up...")
        self.running = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        if self.pyaudio:
            self.pyaudio.terminate()
    
    def start(self):
        """Start the ultra-low latency voice changer"""
        print("🚀 Starting Ultra-Low Latency Voice Changer...")
        print("=" * 60)
        print("🎯 Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            # Setup audio
            self.setup_audio()
            
            # Start all threads
            threads = [
                threading.Thread(target=self.capture_audio, daemon=True, name="Capture"),
                threading.Thread(target=self.process_audio, daemon=True, name="Process"),
                threading.Thread(target=self.play_audio, daemon=True, name="Play"),
                threading.Thread(target=self.monitor_performance, daemon=True, name="Monitor")
            ]
            
            for thread in threads:
                thread.start()
                print(f"✅ Started {thread.name} thread")
            
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice changer...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
            print("👋 Voice changer stopped")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID")
    
    if not api_key or api_key == "your_api_key_here":
        print("❌ Please set your ELEVENLABS_API_KEY in the .env file")
        print("📝 Copy env_template.txt to .env and add your API key")
        sys.exit(1)
    
    if not voice_id or voice_id == "your_voice_id_here":
        print("❌ Please set your VOICE_ID in the .env file")
        print("📝 You can find voice IDs in your ElevenLabs dashboard")
        sys.exit(1)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start voice changer
    voice_changer = UltraLowLatencyVoiceChanger(api_key, voice_id)
    voice_changer.start()

if __name__ == "__main__":
    main() 