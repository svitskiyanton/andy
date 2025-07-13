#!/usr/bin/env python3
"""
STS Streaming Pipeline DEBUG (from Chunks)
=========================================

- Takes input chunks directly from the 'voice_chunks' folder (each chunk as a separate file)
- Sends each chunk to the STS API, applies delays, and plays back the result
- All other logic (delays, playback, etc.) is the same as sts_streaming_pipeline_debug.py

Dependencies:
  pip install pydub pyaudio numpy requests webrtcvad-wheels
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
import numpy as np
import requests
from pydub import AudioSegment
from pydub.playback import play
import atexit
from datetime import datetime
import io

# Global variables for cleanup
running = True
output_chunks = []

def signal_handler(signum, frame):
    global running
    print("\n⏹️  Received interrupt signal, exiting...")
    running = False

def cleanup_handler():
    global running, output_chunks
    if running:
        running = False
        print("💾 Saving output on exit...")
        if output_chunks:
            save_audio_chunks(output_chunks, f"sts_streaming_debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")

def save_audio_chunks(chunks, output_file):
    if not chunks:
        print("⚠️ No audio chunks to save")
        return
    try:
        combined_audio = b''.join(chunks)
        with open(output_file, 'wb') as f:
            f.write(combined_audio)
        print(f"✅ All audio saved to: {output_file}")
        print(f"📊 Total audio size: {len(combined_audio)} bytes")
    except Exception as e:
        print(f"❌ Error saving audio: {e}")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_handler)

class STSStreamingPipelineDebugFromChunks:
    def __init__(self, chunks_dir="voice_chunks"):
        self.chunks_dir = chunks_dir
        self.chunk_files = sorted([os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith('.wav') or f.endswith('.mp3')])
        self.OUTPUT_FILE = f"sts_streaming_debug_output_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
        self.api_key = self._get_api_key()
        self.voice_id = self._get_voice_id()
        self.model_id = "eleven_multilingual_sts_v2"
        self.voice_settings = {
            "stability": 0.8,
            "similarity_boost": 0.85,
            "style": 0.2,
            "use_speaker_boost": True
        }
        self.output_format = "mp3_44100_192"
        self.optimize_streaming_latency = 3
        self.VAD_DELAY = 0.5
        self.STS_DELAY = 1.0
        self.PLAYBACK_DELAY = 0.2
        print(f"🎵 STS Streaming Pipeline DEBUG (from Chunks) Configuration:")
        print(f"   Chunks Dir: {self.chunks_dir}")
        print(f"   Output File: {self.OUTPUT_FILE}")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Voice Settings: {self.voice_settings}")
        print(f"   Output Format: {self.output_format}")
        print(f"   DEBUG Delays: VAD={self.VAD_DELAY}s, STS={self.STS_DELAY}s, Playback={self.PLAYBACK_DELAY}s")

    def _get_api_key(self):
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

    def _get_voice_id(self):
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
        return "GN4wbsbejSnGSa1AzjH5"

    def process_chunks(self):
        global running, output_chunks
        for idx, chunk_file in enumerate(self.chunk_files, 1):
            if not running:
                break
            print(f"\n🎤 Processing chunk {idx}: {chunk_file}")
            # --- VAD delay ---
            time.sleep(self.VAD_DELAY)
            # Load chunk
            if chunk_file.endswith('.mp3'):
                audio_segment = AudioSegment.from_mp3(chunk_file)
            else:
                audio_segment = AudioSegment.from_wav(chunk_file)
            # Export to WAV for API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_wav_path = temp_file.name
            try:
                audio_segment.export(temp_wav_path, format="wav")
                # --- STS API call ---
                headers = {"xi-api-key": self.api_key}
                with open(temp_wav_path, "rb") as audio_file:
                    files = {
                        "audio": ("audio.wav", audio_file, "audio/wav"),
                        "model_id": (None, self.model_id),
                        "remove_background_noise": (None, "false"),
                        "optimize_streaming_latency": (None, "false"),
                        "output_format": (None, self.output_format),
                        "voice_settings": (None, json.dumps(self.voice_settings))
                    }
                    print(f"🎵 Sending chunk {idx} to STS API...")
                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                        headers=headers,
                        files=files,
                        timeout=30
                    )
                    if response.status_code == 200:
                        audio_data = response.content
                        print(f"✅ STS: Received {len(audio_data)} bytes for chunk {idx}")
                        # --- STS delay ---
                        time.sleep(self.STS_DELAY)
                        # --- Playback delay ---
                        time.sleep(self.PLAYBACK_DELAY)
                        # Play audio
                        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                        print(f"🔊 Playing audio for chunk {idx}...")
                        play(audio_segment)
                        print(f"✅ Playback complete for chunk {idx}")
                        output_chunks.append(audio_data)
                    else:
                        print(f"❌ STS API error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Error processing chunk {idx}: {e}")
            finally:
                if os.path.exists(temp_wav_path):
                    try:
                        os.unlink(temp_wav_path)
                    except Exception:
                        pass

if __name__ == "__main__":
    pipeline = STSStreamingPipelineDebugFromChunks()
    pipeline.process_chunks() 