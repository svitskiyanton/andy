#!/usr/bin/env python3
"""
Test Voice Changer with Specific Chunks
Captures 1 chunk of 3 seconds, then 2 chunks of 3 seconds each
Saves each result to separate files for testing
"""

import os
import sys
import time
import threading
import queue
import signal
from dotenv import load_dotenv
import elevenlabs
import pyaudio
import requests
import wave
import io
import numpy as np

class TestVoiceChanger:
    def __init__(self, api_key: str, voice_id: str):
        """
        Initialize the test voice changer
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Target voice ID for transformation
        """
        self.api_key = api_key
        self.voice_id = voice_id
        
        # Set API key for ElevenLabs
        elevenlabs.set_api_key(api_key)
        
        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 2048
        self.test_duration = 3.0  # 3 seconds per test
        
        # Audio objects
        self.pyaudio = None
        self.input_stream = None
        
        # Test tracking
        self.current_test = 0
        self.test_results = []
        
    def setup_audio(self):
        """Setup audio input stream"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Get default devices
            input_device = self.pyaudio.get_default_input_device_info()
            
            print(f"🎤 Input Device: {input_device['name']}")
            print(f"📊 Sample Rate: {self.rate}Hz, Channels: {self.channels}")
            print(f"📦 Chunk Size: {self.chunk_size} samples ({self.chunk_size/self.rate*1000:.0f}ms)")
            print(f"⏱️ Test Duration: {self.test_duration}s per chunk")
            print(f"🇷🇺 Language: Russian")
            
            # Setup input stream
            self.input_stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("✅ Audio setup complete")
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            sys.exit(1)
    
    def pcm_to_wav_bytes(self, pcm_data, sample_rate=16000, channels=1, sample_width=2):
        """Convert PCM data to WAV format"""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buffer.getvalue()
    
    def process_audio_chunk(self, audio_data, chunk_id):
        """Process a single audio chunk through ElevenLabs API"""
        print(f"🔄 Processing chunk {chunk_id}...")
        
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream"
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Parameters optimized for Russian language
        params = {
            "optimize_streaming_latency": 4,
            "output_format": "pcm_16000",
            "model_id": "eleven_multilingual_sts_v2",
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 0.7
            }
        }
        
        try:
            # Convert to WAV
            wav_data = self.pcm_to_wav_bytes(audio_data)
            
            # Prepare files for upload
            files = {
                'audio': (f'chunk_{chunk_id}.wav', wav_data, 'audio/wav')
            }
            
            # Send to ElevenLabs API
            response = requests.post(
                url,
                headers=headers,
                params=params,
                files=files,
                stream=True,
                timeout=15
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
                    
                    # Save to file
                    output_filename = f"test_chunk_{chunk_id}.wav"
                    with wave.open(output_filename, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.rate)
                        wf.writeframes(combined_audio)
                    
                    print(f"✅ Chunk {chunk_id} saved to {output_filename}")
                    return output_filename
                else:
                    print(f"❌ No audio data received for chunk {chunk_id}")
                    return None
                    
            else:
                print(f"❌ API Error for chunk {chunk_id}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_id}: {e}")
            return None
    
    def capture_and_process_chunk(self, chunk_id):
        """Capture a single chunk of audio and process it"""
        print(f"\n🎤 Capturing chunk {chunk_id} ({self.test_duration}s)...")
        print("🎯 Speak now! Press Enter when done...")
        
        # Wait for user to start
        input()
        
        # Capture audio for specified duration
        audio_chunks = []
        start_time = time.time()
        
        while time.time() - start_time < self.test_duration:
            try:
                audio_data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunks.append(audio_data)
            except Exception as e:
                print(f"⚠️ Audio capture error: {e}")
                break
        
        # Combine all chunks
        combined_audio = b''.join(audio_chunks)
        actual_duration = len(audio_chunks) * self.chunk_size / self.rate
        print(f"📦 Captured {actual_duration:.1f}s of audio")
        
        # Process the chunk
        result_file = self.process_audio_chunk(combined_audio, chunk_id)
        if result_file:
            self.test_results.append(result_file)
        
        return result_file
    
    def run_tests(self):
        """Run the test sequence"""
        print("🧪 Voice Changer Test Suite")
        print("=" * 50)
        print("Test: 3 chunks of 3 seconds each")
        print("=" * 50)
        
        try:
            # Setup audio
            self.setup_audio()
            
            # Test: 3 chunks of 3 seconds each
            print(f"\n{'='*20} TEST - 3 CHUNKS {'='*20}")
            result1 = self.capture_and_process_chunk(1)
            result2 = self.capture_and_process_chunk(2)
            result3 = self.capture_and_process_chunk(3)
            
            # Summary
            print(f"\n{'='*50}")
            print("📊 Test Results Summary:")
            print(f"{'='*50}")
            
            if result1:
                print(f"✅ Chunk 1: {result1}")
            else:
                print(f"❌ Chunk 1: Failed")
                
            if result2:
                print(f"✅ Chunk 2: {result2}")
            else:
                print(f"❌ Chunk 2: Failed")
                
            if result3:
                print(f"✅ Chunk 3: {result3}")
            else:
                print(f"❌ Chunk 3: Failed")
            
            print(f"\n📁 All test files saved in current directory")
            
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
        except Exception as e:
            print(f"❌ Test error: {e}")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.pyaudio:
                self.pyaudio.terminate()

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
    
    # Create and run tests
    tester = TestVoiceChanger(api_key, voice_id)
    tester.run_tests()

if __name__ == "__main__":
    main() 