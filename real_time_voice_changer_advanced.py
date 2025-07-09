#!/usr/bin/env python3
"""
Advanced Real-Time Voice Changer using ElevenLabs Speech-to-Speech API
Features parallel processing, multiple API workers, and optimized buffering
"""

import os
import sys
import time
import threading
import queue
import signal
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import elevenlabs
import pyaudio
import requests
import wave
import io
import numpy as np

class AdvancedVoiceChanger:
    def __init__(self, api_key: str, voice_id: str, num_workers: int = 2):
        """
        Initialize the advanced real-time voice changer
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Target voice ID for transformation
            num_workers: Number of parallel API workers
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.num_workers = num_workers
        
        # Set API key for ElevenLabs
        elevenlabs.set_api_key(api_key)
        
        # Audio processing queues
        self.input_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=50)
        self.worker_queues = [queue.Queue(maxsize=5) for _ in range(num_workers)]
        
        # Control flags
        self.running = True
        
        # Audio settings - optimized for low latency
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 2048  # Smaller chunks for lower latency (128ms)
        self.buffer_duration = 1.0  # 1 second of audio per API call
        
        # Audio objects
        self.pyaudio = None
        self.input_stream = None
        self.output_stream = None
        
        # Statistics
        self.processed_chunks = 0
        self.start_time = None
        self.latency_history = collections.deque(maxlen=20)
        self.worker_stats = [0] * num_workers
        
        # Audio buffer for batching
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_id = 0
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
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
            print(f"⚡ Workers: {self.num_workers}")
            
            # Setup input stream
            self.input_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Setup output stream
            self.output_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
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
    
    def process_audio_worker(self, worker_id: int):
        """Worker function for processing audio chunks"""
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream"
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Optimized parameters for low latency
        params = {
            "optimize_streaming_latency": 4,  # Maximum optimizations
            "output_format": "pcm_16000",
            "model_id": "eleven_multilingual_sts_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        while self.running:
            try:
                # Get audio from worker queue
                if not self.worker_queues[worker_id].empty():
                    buffer_id, audio_buffer = self.worker_queues[worker_id].get(timeout=0.1)
                    start_time = time.time()
                    
                    # Convert to WAV
                    wav_data = self.pcm_to_wav_bytes(audio_buffer)
                    
                    # Prepare files for upload
                    files = {
                        'audio': (f'audio_{worker_id}.wav', wav_data, 'audio/wav')
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
                        for chunk in response.iter_content(chunk_size=8192):
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
                                    self.output_queue.put((buffer_id, chunk))
                            
                            # Calculate latency
                            end_time = time.time()
                            latency = (end_time - start_time) * 1000
                            self.latency_history.append(latency)
                            self.worker_stats[worker_id] += 1
                            
                    else:
                        print(f"❌ API Error (Worker {worker_id}): {response.status_code} - {response.text}")
                        
            except queue.Empty:
                continue
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error (Worker {worker_id}): {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Processing error (Worker {worker_id}): {e}")
                time.sleep(0.1)
    
    def process_audio(self):
        """Distribute audio processing across workers"""
        print("🔄 Starting voice transformation...")
        
        # Start worker threads
        worker_threads = []
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self.process_audio_worker, 
                args=(i,), 
                daemon=True, 
                name=f"Worker-{i}"
            )
            thread.start()
            worker_threads.append(thread)
            print(f"✅ Started Worker-{i} thread")
        
        # Distribute work
        worker_index = 0
        while self.running:
            try:
                if not self.input_queue.empty():
                    buffer_id, audio_buffer = self.input_queue.get(timeout=0.1)
                    
                    # Round-robin distribution
                    self.worker_queues[worker_index].put((buffer_id, audio_buffer))
                    worker_index = (worker_index + 1) % self.num_workers
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Distribution error: {e}")
                time.sleep(0.01)
    
    def play_audio(self):
        """Play transformed audio through speakers"""
        print("🔊 Starting audio output...")
        
        # Buffer for ordered playback
        audio_buffer = {}
        next_buffer_id = 0
        
        try:
            while self.running:
                try:
                    if not self.output_queue.empty():
                        buffer_id, audio_data = self.output_queue.get(timeout=0.1)
                        
                        # Store in buffer
                        if buffer_id not in audio_buffer:
                            audio_buffer[buffer_id] = []
                        audio_buffer[buffer_id].append(audio_data)
                        
                        # Play complete buffers in order
                        while next_buffer_id in audio_buffer:
                            for chunk in audio_buffer[next_buffer_id]:
                                self.output_stream.write(chunk)
                            del audio_buffer[next_buffer_id]
                            next_buffer_id += 1
                            self.processed_chunks += 1
                            
                    else:
                        time.sleep(0.001)
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
                time.sleep(2)  # Update every 2 seconds
                
                if self.start_time and self.latency_history:
                    elapsed = time.time() - self.start_time
                    chunks_per_second = self.processed_chunks / elapsed if elapsed > 0 else 0
                    avg_latency = sum(self.latency_history) / len(self.latency_history)
                    
                    print(f"📊 Stats: {self.processed_chunks} chunks processed, "
                          f"{chunks_per_second:.1f} chunks/sec, "
                          f"Avg Latency: {avg_latency:.0f}ms")
                    
                    # Worker statistics
                    worker_stats_str = ", ".join([f"W{i}:{stats}" for i, stats in enumerate(self.worker_stats)])
                    print(f"⚡ Workers: {worker_stats_str}")
                    
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
        
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def start(self):
        """Start the advanced real-time voice changer"""
        print("🚀 Starting Advanced Real-Time Voice Changer...")
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
    voice_changer = AdvancedVoiceChanger(api_key, voice_id, num_workers=2)
    voice_changer.start()

if __name__ == "__main__":
    main() 