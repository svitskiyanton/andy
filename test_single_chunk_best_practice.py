#!/usr/bin/env python3
"""
Best Practice Single Chunk Voice Changer Test (ElevenLabs SDK, batch endpoint, mp3 output)
"""

import os
import sys
import time
import signal
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
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

class BestPracticeSingleChunkTest:
    def __init__(self, api_key: str, voice_id: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.elevenlabs = ElevenLabs(api_key=api_key)
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

    def process_audio(self, wav_data):
        print("🔄 Processing audio through ElevenLabs API (batch endpoint)...")
        audio_stream = io.BytesIO(wav_data)
        start_time = time.time()
        result = self.elevenlabs.speech_to_speech.convert(
            voice_id=self.voice_id,
            audio=audio_stream,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT,
            voice_settings={
                "stability": STABILITY,
                "similarity_boost": SIMILARITY_BOOST
            }
        )
        end_time = time.time()
        print(f"✅ Processing completed in {(end_time - start_time)*1000:.0f}ms")
        return result

    def run_test(self):
        print("🧪 Best Practice Single Chunk Voice Changer Test")
        print("=" * 60)
        try:
            self.setup_audio()
            pcm_data = self.capture_audio()
            self.save_wav("best_practice_original.wav", pcm_data)
            # Convert PCM to WAV bytes for API
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                wf.writeframes(pcm_data)
            wav_bytes.seek(0)
            # Process with ElevenLabs
            result_mp3 = self.process_audio(wav_bytes.read())
            with open("best_practice_result.mp3", "wb") as f:
                f.write(result_mp3)
            print(f"📁 Transformed: best_practice_result.mp3")
            print("\n🎵 You can now compare the original WAV and transformed MP3!")
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
    tester = BestPracticeSingleChunkTest(api_key, voice_id)
    tester.run_test()

if __name__ == "__main__":
    main() 