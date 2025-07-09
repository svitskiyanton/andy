#!/usr/bin/env python3
"""
Direct API Single Chunk Voice Changer Test (no SDK, best practices for Russian)
"""

import os
import sys
import time
import signal
import requests
from dotenv import load_dotenv
import pyaudio
import wave
import io

# === CONFIGURABLE SETTINGS ===
STABILITY = 0.7
SIMILARITY_BOOST = 0.3
OUTPUT_FORMAT = "mp3_44100_128"  # Best quality for ElevenLabs
MODEL_ID = "eleven_multilingual_sts_v2"  # For Russian
CHUNK_SIZE = 2048
RATE = 16000
CHANNELS = 1
DURATION = 3.0  # seconds

class DirectAPISingleChunkTest:
    def __init__(self, api_key: str, voice_id: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.audio_format = pyaudio.paInt16
        self.channels = CHANNELS
        self.rate = RATE
        self.chunk_size = CHUNK_SIZE
        self.test_duration = DURATION
        self.pyaudio = None
        self.input_stream = None

    def setup_audio(self):
        try:
            self.pyaudio = pyaudio.PyAudio()
            input_device = self.pyaudio.get_default_input_device_info()
            print(f"🎤 Input Device: {input_device['name']}")
            print(f"📊 Sample Rate: {self.rate}Hz, Channels: {self.channels}")
            print(f"📦 Chunk Size: {self.chunk_size} samples ({self.chunk_size/self.rate*1000:.0f}ms)")
            print(f"⏱️ Test Duration: {self.test_duration}s")
            print(f"🇷🇺 Model: {MODEL_ID}")
            print(f"🎯 Voice ID: {self.voice_id}")
            print(f"🎚️ Stability: {STABILITY}, Similarity Boost: {SIMILARITY_BOOST}")
            print(f"🎵 Output Format: {OUTPUT_FORMAT}")
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

    def capture_audio(self):
        print(f"\n🎤 Capturing {self.test_duration}s of audio...")
        print("🎯 Speak now! Press Enter to start recording...")
        input()
        print("🔴 Recording... (3 seconds)")
        audio_chunks = []
        start_time = time.time()
        while time.time() - start_time < self.test_duration:
            try:
                audio_data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunks.append(audio_data)
            except Exception as e:
                print(f"⚠️ Audio capture error: {e}")
                break
        combined_audio = b''.join(audio_chunks)
        actual_duration = len(audio_chunks) * self.chunk_size / self.rate
        print(f"📦 Captured {actual_duration:.1f}s of audio")
        return combined_audio

    def save_wav(self, filename, pcm_data):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(pcm_data)
        print(f"📁 Saved WAV: {filename}")

    def pcm_to_wav_bytes(self, pcm_data):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(pcm_data)
        return buffer.getvalue()

    def process_audio(self, pcm_data):
        print("🔄 Processing audio through ElevenLabs API (direct call)...")
        
        # Convert PCM to WAV
        wav_data = self.pcm_to_wav_bytes(pcm_data)
        
        # API endpoint (non-streaming for best quality)
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}"
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Best practice parameters for Russian
        params = {
            "model_id": MODEL_ID,
            "output_format": OUTPUT_FORMAT,
            "voice_settings": {
                "stability": STABILITY,
                "similarity_boost": SIMILARITY_BOOST
            }
        }
        
        files = {
            'audio': ('audio.wav', wav_data, 'audio/wav')
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                files=files,
                timeout=30
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                print(f"✅ Processing completed in {processing_time:.0f}ms")
                return response.content
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return None

    def run_test(self):
        print("🧪 Direct API Single Chunk Voice Changer Test")
        print("=" * 60)
        print("🔧 Using direct API calls (no SDK)")
        print("=" * 60)
        
        try:
            self.setup_audio()
            pcm_data = self.capture_audio()
            self.save_wav("direct_api_original.wav", pcm_data)
            
            # Process with ElevenLabs
            result_mp3 = self.process_audio(pcm_data)
            
            if result_mp3:
                with open("direct_api_result.mp3", "wb") as f:
                    f.write(result_mp3)
                print(f"📁 Transformed: direct_api_result.mp3")
                print("\n🎵 You can now compare the original WAV and transformed MP3!")
                print("🔧 This version uses direct API calls with best practices for Russian")
            else:
                print("❌ Processing failed")
                
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        except Exception as e:
            print(f"❌ Test error: {e}")
        finally:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            if self.pyaudio:
                self.pyaudio.terminate()

def signal_handler(signum, frame):
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

def main():
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID")
    
    if not api_key or api_key == "your_api_key_here":
        print("❌ Please set your ELEVENLABS_API_KEY in the .env file")
        sys.exit(1)
    
    if not voice_id or voice_id == "your_voice_id_here":
        print("❌ Please set your VOICE_ID in the .env file")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    tester = DirectAPISingleChunkTest(api_key, voice_id)
    tester.run_test()

if __name__ == "__main__":
    main() 