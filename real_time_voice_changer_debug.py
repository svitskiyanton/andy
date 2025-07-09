#!/usr/bin/env python3
"""
Debug Real-Time Voice Changer using ElevenLabs Speech-to-Speech API
Writes output to local file for testing and debugging
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

class DebugVoiceChanger:
    def __init__(self, api_key: str, voice_id: str):
        """
        Initialize the debug voice changer
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Target voice ID for transformation
        """
        self.api_key = api_key
        self.voice_id = voice_id
        
        # Set API key for ElevenLabs
        elevenlabs.set_api_key(api_key)
        
        # Audio processing queues
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=20)
        
        # Control flags
        self.running = True
        
        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 2048
        self.buffer_duration = 1.0  # 1 second buffers
        
        # Audio objects
        self.pyaudio = None
        self.input_stream = None
        
        # Statistics
        self.processed_chunks = 0
        self.start_time = None
        self.latency_history = collections.deque(maxlen=10)
        
        # Audio buffer for batching
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_id = 0
        
        # Output file for debugging
        self.output_filename = "debug_output.wav"
        self.output_wav = None
        self.output_lock = threading.Lock()
        
        # Performance tracking
        self.last_process_time = 0
        self.min_interval = 0.5
        
    def setup_audio(self):
        """Setup audio input stream and output file"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Get default devices
            input_device = self.pyaudio.get_default_input_device_info()
            
            print(f"🎤 Input Device: {input_device['name']}")
            print(f"📊 Sample Rate: {self.rate}Hz, Channels: {self.channels}")
            print(f"📦 Chunk Size: {self.chunk_size} samples ({self.chunk_size/self.rate*1000:.0f}ms)")
            print(f"🔄 Buffer Duration: {self.buffer_duration}s")
            print(f"📁 Output File: {self.output_filename}")
            print(f"🇷🇺 Language: Russian")
            
            # Setup input stream
            self.input_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Setup output WAV file
            self.output_wav = wave.open(self.output_filename, 'wb')
            self.output_wav.setnchannels(self.channels)
            self.output_wav.setsampwidth(2)  # 16-bit
            self.output_wav.setframerate(self.rate)
            
            print("✅ Audio setup complete")
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            sys.exit(1)
    
    def capture_audio(self):
        """Capture audio from microphone"""
        print("🎤 Starting audio capture...")
        
        try:
            while self.running:
                try:
                    # Read audio data
                    audio_data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Check if audio has actual content (not just silence)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    if np.abs(audio_array).mean() > 100:  # Threshold for non-silence
                        # Add to buffer
                        with self.buffer_lock:
                            self.audio_buffer.append(audio_data)
                            
                            # Check if we have enough audio for processing
                            buffer_samples = len(self.audio_buffer) * self.chunk_size
                            buffer_duration = buffer_samples / self.rate
                            
                            if buffer_duration >= self.buffer_duration:
                                # Combine all chunks into one buffer
                                combined_audio = b''.join(self.audio_buffer)
                                self.input_queue.put((self.buffer_id, combined_audio))
                                self.buffer_id += 1
                                self.audio_buffer = []  # Clear buffer
                                print(f"📦 Sent buffer {self.buffer_id-1} ({buffer_duration:.1f}s)")
                    else:
                        # For silence, add a small delay
                        time.sleep(0.01)
                            
                except Exception as e:
                    print(f"⚠️ Audio capture error: {e}")
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\n🛑 Audio capture interrupted")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
    
    def pcm_to_wav_bytes(self, pcm_data, sample_rate=16000, channels=1, sample_width=2):
        """Convert PCM data to WAV format"""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buffer.getvalue()
    
    def process_audio(self):
        """Process audio through ElevenLabs Speech-to-Speech API"""
        print("🔄 Starting voice transformation...")
        
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream"
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Parameters optimized for Russian language
        params = {
            "optimize_streaming_latency": 4,  # Maximum optimizations
            "output_format": "pcm_16000",
            "model_id": "eleven_multilingual_sts_v2",  # Supports Russian
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 0.7
            }
        }
        
        while self.running:
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_process_time < self.min_interval:
                    time.sleep(0.01)
                    continue
                
                if not self.input_queue.empty():
                    start_time = time.time()
                    self.last_process_time = start_time
                    
                    # Get audio buffer with ID for ordering
                    buffer_id, audio_buffer = self.input_queue.get(timeout=0.1)
                    print(f"🔄 Processing buffer {buffer_id}...")
                    
                    # Convert to WAV
                    wav_data = self.pcm_to_wav_bytes(audio_buffer)
                    
                    # Prepare files for upload
                    files = {
                        'audio': (f'audio_{buffer_id}.wav', wav_data, 'audio/wav')
                    }
                    
                    # Send to ElevenLabs API
                    response = requests.post(
                        url,
                        headers=headers,
                        params=params,
                        files=files,
                        stream=True,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        # Collect all audio data
                        audio_chunks = []
                        for chunk in response.iter_content(chunk_size=4096):
                            if chunk:
                                audio_chunks.append(chunk)
                        
                        if audio_chunks:
                            # Combine all chunks
                            combined_audio = b''.join(audio_chunks)
                            
                            # Put in output queue with buffer ID for ordering
                            self.output_queue.put((buffer_id, combined_audio))
                            
                            # Calculate latency
                            end_time = time.time()
                            latency = (end_time - start_time) * 1000
                            self.latency_history.append(latency)
                            self.processed_chunks += 1
                            
                            print(f"✅ Buffer {buffer_id} processed in {latency:.0f}ms")
                            
                    else:
                        print(f"❌ API Error for buffer {buffer_id}: {response.status_code} - {response.text}")
                        
            except queue.Empty:
                continue
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                time.sleep(0.1)
    
    def write_audio_to_file(self):
        """Write transformed audio to WAV file in proper order"""
        print("📝 Starting audio file writing...")
        
        # Buffer for ordered chunks
        audio_buffer = {}
        next_buffer_id = 0
        
        try:
            while self.running:
                try:
                    if not self.output_queue.empty():
                        buffer_id, audio_data = self.output_queue.get(timeout=0.1)
                        
                        # Store in buffer
                        audio_buffer[buffer_id] = audio_data
                        print(f"📥 Received buffer {buffer_id}, waiting for {next_buffer_id}")
                        
                        # Write complete buffers in order
                        while next_buffer_id in audio_buffer:
                            with self.output_lock:
                                self.output_wav.writeframes(audio_buffer[next_buffer_id])
                                print(f"📝 Wrote buffer {next_buffer_id} to file")
                            
                            del audio_buffer[next_buffer_id]
                            next_buffer_id += 1
                            
                    else:
                        time.sleep(0.001)
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n🛑 Audio writing interrupted")
        except Exception as e:
            print(f"❌ Audio writing error: {e}")
        finally:
            # Write any remaining buffers
            for buffer_id in sorted(audio_buffer.keys()):
                if buffer_id >= next_buffer_id:
                    with self.output_lock:
                        self.output_wav.writeframes(audio_buffer[buffer_id])
                        print(f"📝 Wrote remaining buffer {buffer_id} to file")
    
    def monitor_performance(self):
        """Monitor and display performance statistics"""
        self.start_time = time.time()
        
        while self.running:
            try:
                time.sleep(2)
                
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
            
        if self.output_wav:
            self.output_wav.close()
            
        if self.pyaudio:
            self.pyaudio.terminate()
        
        print(f"✅ Output saved to: {self.output_filename}")
    
    def start(self):
        """Start the debug voice changer"""
        print("🚀 Starting Debug Voice Changer...")
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
                threading.Thread(target=self.write_audio_to_file, daemon=True, name="Write"),
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
    voice_changer = DebugVoiceChanger(api_key, voice_id)
    voice_changer.start()

if __name__ == "__main__":
    main() 