#!/usr/bin/env python3
"""
Real-time Microphone to STS Streaming Pipeline
=============================================

- Listens to microphone, splits speech into natural phrases (VAD + silence detection)
- For each phrase, sends audio to STS API, plays back the result
- Keeps debug delays (VAD, STS, playback)
- Does NOT require saving input/output files by default

Dependencies:
  pip install pyaudio numpy webrtcvad-wheels requests pydub
"""

import os
import sys
import time
import signal
import atexit
import threading
import queue
from datetime import datetime
import numpy as np
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import json
import io

# Try to import webrtcvad-wheels first, fallback to webrtcvad
try:
    import webrtcvad_wheels as webrtcvad
    print("✓ Using webrtcvad-wheels for VAD")
except ImportError:
    try:
        import webrtcvad
        print("✓ Using webrtcvad for VAD")
    except ImportError:
        print("✗ No VAD library found. Please install webrtcvad-wheels")
        sys.exit(1)

# --- STS API Settings (from sts_streaming_pipeline_debug.py) ---
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "GN4wbsbejSnGSa1AzjH5")
MODEL = "eleven_multilingual_sts_v2"
OUTPUT_FORMAT = "mp3_44100_192"
VOICE_SETTINGS = {
    "stability": 0.8,
    "similarity_boost": 0.85,
    "style": 0.2,
    "use_speaker_boost": True
}

def get_api_key():
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
            with open(".env", "a") as f:
                f.write(f"ELEVENLABS_API_KEY={api_key}\n")
            print("✅ API key saved to .env file for future use")
        except Exception:
            print("⚠️ Could not save API key to file")
        return api_key
    return None

API_KEY = get_api_key()

# --- Delays (debug, from sts_streaming_pipeline_debug.py) ---
VAD_DELAY = 0.5      # 500ms after chunking
STS_DELAY = 1.0      # 1s after STS processing
PLAYBACK_DELAY = 0.2 # 200ms before playback

# --- Audio/VAD Settings (from simple_realtime_voice_chunker.py) ---
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
CHANNELS = 1
FORMAT = pyaudio.paInt16
VAD_MODE = 2
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 0.03
SILENCE_THRESHOLD = 0.008
SILENCE_DURATION = 0.5
MIN_PHRASE_DURATION = 1.0
MAX_PHRASE_DURATION = 8.0

# --- Main Class ---
class RealtimeMicToSTS:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.is_running = False
        print(f"🎤 Real-time Mic→STS initialized:")
        print(f"   Sample Rate: {SAMPLE_RATE}Hz, VAD Mode: {VAD_MODE}")
        print(f"   Delays: VAD={VAD_DELAY}s, STS={STS_DELAY}s, Playback={PLAYBACK_DELAY}s")

    def detect_speech_in_chunk(self, audio_chunk):
        # Resample to 16kHz for VAD
        audio_16k = self._resample_audio(audio_chunk, VAD_SAMPLE_RATE)
        audio_int16 = (audio_16k * 32767).astype(np.int16)
        frame_size = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION)
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            if len(frame) == frame_size:
                if self.vad.is_speech(frame.tobytes(), VAD_SAMPLE_RATE):
                    return True
        return False

    def _resample_audio(self, audio_data, target_sample_rate):
        if target_sample_rate == SAMPLE_RATE:
            return audio_data
        original_length = len(audio_data)
        target_length = int(original_length * target_sample_rate / SAMPLE_RATE)
        indices = np.linspace(0, original_length - 1, target_length)
        return np.interp(indices, np.arange(original_length), audio_data)

    def start(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        print("🎤 Listening for speech...")
        self.is_running = True
        try:
            self._listen_loop()
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.stop()

    def _listen_loop(self):
        audio_buffer = b""
        silence_start = None
        phrase_start_time = time.time()
        phrase_count = 0
        while self.is_running:
            data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_buffer += data
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
            current_time = time.time()
            phrase_duration = current_time - phrase_start_time
            is_speech = self.detect_speech_in_chunk(audio_data)
            if audio_level < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = current_time
            else:
                silence_start = None
            should_end = False
            end_reason = ""
            if (len(audio_buffer) > 0 and phrase_duration >= MIN_PHRASE_DURATION and silence_start is not None and (current_time - silence_start) >= SILENCE_DURATION):
                should_end = True
                end_reason = f"silence detected for {current_time - silence_start:.1f}s"
            elif phrase_duration >= MAX_PHRASE_DURATION and len(audio_buffer) > 0:
                should_end = True
                end_reason = f"max duration reached ({MAX_PHRASE_DURATION}s)"
            if should_end:
                phrase_count += 1
                duration = len(audio_buffer) / (SAMPLE_RATE * 2)
                print(f"🎤 Phrase {phrase_count} detected: {duration:.2f}s ({end_reason})")
                # --- VAD delay ---
                time.sleep(VAD_DELAY)
                # Send to STS
                self.process_chunk_with_sts(audio_buffer, phrase_count)
                # Reset for next phrase
                audio_buffer = b""
                silence_start = None
                phrase_start_time = time.time()
            time.sleep(0.01)

    def process_chunk_with_sts(self, audio_buffer, phrase_num):
        # Convert to WAV for API
        import os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav_path = temp_file.name
        try:
            audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=SAMPLE_RATE,
                sample_width=2,
                channels=1
            )
            audio_segment.export(temp_wav_path, format="wav")
            # --- STS API call ---
            try:
                headers = {"xi-api-key": API_KEY}
                with open(temp_wav_path, "rb") as audio_file:
                    files = {
                        "audio": ("audio.wav", audio_file, "audio/wav"),
                        "model_id": (None, MODEL),
                        "remove_background_noise": (None, "false"),
                        "optimize_streaming_latency": (None, "false"),
                        "output_format": (None, OUTPUT_FORMAT),
                        "voice_settings": (None, json.dumps(VOICE_SETTINGS))
                    }
                    print(f"🎵 Sending phrase {phrase_num} to STS API...")
                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/speech-to-speech/{VOICE_ID}/stream",
                        headers=headers,
                        files=files,
                        timeout=30
                    )
                    if response.status_code == 200:
                        audio_data = response.content
                        print(f"✅ STS: Received {len(audio_data)} bytes for phrase {phrase_num}")
                        # --- STS delay ---
                        time.sleep(STS_DELAY)
                        # --- Playback delay ---
                        time.sleep(PLAYBACK_DELAY)
                        # Play audio
                        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                        print(f"🔊 Playing audio for phrase {phrase_num}...")
                        play(audio_segment)
                        print(f"✅ Playback complete for phrase {phrase_num}")
                    else:
                        print(f"❌ STS API error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ STS processing error: {e}")
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        print("🛑 Stopped listening and cleaned up.")

if __name__ == "__main__":
    pipeline = RealtimeMicToSTS()
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal, stopping...")
        pipeline.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(pipeline.stop)
    pipeline.start() 