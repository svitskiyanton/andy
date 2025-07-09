#!/usr/bin/env python3
"""
Real-time Voice Changer with Smooth Live Playback (Pro Version)
Upgraded for ElevenLabs Pro limits and Turbo TTS model
"""

# --- CODE COPIED FROM realtime_voice_changer_smooth.py ---
# (Full code pasted here, ready for Pro modifications)

import asyncio
import json
import base64
import time
import numpy as np
import pyaudio
from collections import deque
import threading
import queue
from pydub import AudioSegment
import io
import os
import subprocess
import sys
from google.cloud import speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealtimeVoiceChangerSmoothPro:
    def __init__(self):
        # Audio settings
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        
        # Buffer settings (Pro limits)
        self.BUFFER_SIZE = 30  # Increased for Pro
        self.MIN_BUFFER_SIZE = 10
        self.MAX_BUFFER_SIZE = 60
        
        # Audio objects
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=200)
        self.playback_thread = None
        
        # Playback control
        self.playback_started = False
        self.streaming_complete = False
        self.mp3_chunks = []
        
        # Output file for all TTS audio
        self.all_audio_chunks = []
        
        # Output file
        self.OUTPUT_FILE = "realtime_voice_changer_output_pro.mp3"
        
        # In-memory LIFO queue for STT → TTS communication
        self.text_queue = queue.Queue(maxsize=20)  # Increased for Pro
        
        # File logging for testing
        self.QUEUE_LOG_FILE = "queue_log_pro.txt"
        self.OUTPUT_MP3_FILE = "realtime_voice_changer_output_pro.mp3"
        
        # Control
        self.running = True
        
        # STT settings
        self.STT_SAMPLE_RATE = 44100  # Pro supports 44.1kHz
        self.STT_CHANNELS = 1
        self.STT_FORMAT = pyaudio.paInt16
        self.STT_CHUNK_SIZE = 2048  # Larger chunk for Pro
        
        # STT objects
        self.stt_audio = pyaudio.PyAudio()
        self.speech_client = None
        self.init_google_client()
        
        # STT buffers
        self.audio_buffer = []
        self.silence_buffer = []
        self.current_phrase = ""
        self.silence_start = None
        self.max_silence_duration = 1.0  # Lower for Pro
        
        # STT settings
        self.BUFFER_DURATION = 2.0  # Lower for Pro
        self.BUFFER_SIZE_STT = int(self.STT_SAMPLE_RATE * self.BUFFER_DURATION)
        self.SILENCE_DURATION = 0.5  # Lower for Pro
        self.SILENCE_THRESHOLD = 0.008  # Slightly lower for Pro
        
        # Text processing
        self.text_processor_thread = None

    def init_google_client(self):
        try:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                self.speech_client = speech.SpeechClient()
                print("✅ STT: Google Cloud Speech client initialized")
            else:
                print("⚠️  STT: Google Cloud credentials not found")
        except Exception as e:
            print(f"❌ STT: Failed to initialize Google Cloud client: {e}")

    def transcribe_audio(self, audio_chunk):
        if not self.speech_client:
            return None
        try:
            print("🔍 STT: Transcribing continuous speech...")
            audio = speech.RecognitionAudio(content=audio_chunk)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.STT_SAMPLE_RATE,
                language_code="ru-RU",
                enable_automatic_punctuation=True,
                model="latest_long",
                enable_word_time_offsets=False,
                enable_word_confidence=False,
                use_enhanced=True,
                max_alternatives=1
            )
            response = self.speech_client.recognize(config=config, audio=audio)
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            text = transcribed_text.strip()
            if text:
                print(f"📝 STT: Complete phrase transcribed: '{text}'")
                return text
            else:
                print("🔇 STT: No speech detected in phrase")
                return None
        except Exception as e:
            print(f"❌ STT: Transcription error: {e}")
            return None

    def add_text_to_queue(self, phrase):
        try:
            self.text_queue.put_nowait(phrase)
            print(f"📝 STT → TTS Queue: '{phrase}'")
            self.log_to_file(f"ADDED: {phrase}")
        except queue.Full:
            try:
                old_phrase = self.text_queue.get_nowait()
                self.text_queue.put_nowait(phrase)
                print(f"🔄 STT → TTS Queue: Replaced '{old_phrase}' with '{phrase}'")
                self.log_to_file(f"REPLACED: '{old_phrase}' → '{phrase}'")
            except Exception as e:
                print(f"❌ Error managing queue: {e}")
        except Exception as e:
            print(f"❌ Error adding to queue: {e}")

    def log_to_file(self, message):
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.QUEUE_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"❌ Error logging to file: {e}")

    def stt_worker(self):
        print("🎤 STT Worker: Starting continuous streaming recognition...")
        try:
            stream = self.stt_audio.open(
                format=self.STT_FORMAT,
                channels=self.STT_CHANNELS,
                rate=self.STT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.STT_CHUNK_SIZE
            )
            print("✅ STT: Audio input stream started")
            audio_buffer = b""
            silence_start = None
            silence_threshold = self.SILENCE_THRESHOLD
            min_phrase_duration = 1.0
            max_phrase_duration = 8.0
            silence_duration = self.SILENCE_DURATION
            phrase_start_time = time.time()
            while self.running:
                try:
                    data = stream.read(self.STT_CHUNK_SIZE, exception_on_overflow=False)
                    audio_buffer += data
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
                    current_time = time.time()
                    phrase_duration = current_time - phrase_start_time
                    if audio_level < silence_threshold:
                        if silence_start is None:
                            silence_start = current_time
                    else:
                        silence_start = None
                    should_process = False
                    if (len(audio_buffer) > 0 and 
                        phrase_duration >= min_phrase_duration and
                        silence_start is not None and
                        (current_time - silence_start) >= silence_duration):
                        should_process = True
                    elif phrase_duration >= max_phrase_duration and len(audio_buffer) > 0:
                        should_process = True
                    if should_process:
                        print(f"🎤 STT: Processing {len(audio_buffer)} bytes ({phrase_duration:.1f}s) - silence detected")
                        transcribed_text = self.transcribe_audio(audio_buffer)
                        if transcribed_text:
                            print(f"📝 STT: Complete phrase: '{transcribed_text}'")
                            self.add_text_to_queue(transcribed_text)
                        else:
                            print("🔇 STT: No speech detected in phrase")
                        audio_buffer = b""
                        silence_start = None
                        phrase_start_time = current_time
                    time.sleep(0.01)
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            stream.stop_stream()
            stream.close()
            self.stt_audio.terminate()
            print("✅ STT Worker stopped")
        except Exception as e:
            print(f"❌ STT Worker error: {e}")

    def text_processor_worker(self):
        print("📝 Text Processor: Monitoring in-memory queue...")
        while self.running:
            try:
                try:
                    text = self.text_queue.get(timeout=0.1)
                    print(f"🎵 TTS: Processing text from queue: '{text}'")
                    self.log_to_file(f"PROCESSING: {text}")
                    asyncio.run(self.stream_text_to_tts(text))
                    self.text_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                print(f"❌ Text processor error: {e}")
                time.sleep(0.1)

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
                with open(".env", "w") as f:
                    f.write(f"ELEVENLABS_API_KEY={api_key}\n")
                print("✅ API key saved to .env file for future use")
            except Exception:
                print("⚠️ Could not save API key to file")
            return api_key
        return None

    async def stream_text_to_tts(self, text):
        if not text.strip():
            return
        text_chunks = self._split_text_into_chunks(text, max_chunk_size=100)
        import websockets
        # Use the Eleven Multilingual v2 model for balanced quality and speed
        uri = "wss://api.elevenlabs.io/v1/text-to-speech/GN4wbsbejSnGSa1AzjH5/stream-input?model_id=eleven_multilingual_v2&optimize_streaming_latency=4"
        try:
            async with websockets.connect(uri) as websocket:
                api_key = self._get_api_key()
                if not api_key:
                    print("❌ API key not found. Skipping TTS.")
                    return
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    },
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(init_message))
                for chunk in text_chunks:
                    message = {
                        "text": chunk,
                        "xi_api_key": api_key
                    }
                    await websocket.send(json.dumps(message))
                end_message = {
                    "text": "",
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(end_message))
                await self._smooth_audio_streaming(websocket)
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")

    def _split_text_into_chunks(self, text, max_chunk_size=100):
        chunks = []
        current_chunk = ""
        current_length = 0
        import re
        parts = re.split(r'(\s+)', text)
        for part in parts:
            if current_length + len(part) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = part
                current_length = len(part)
            else:
                current_chunk += part
                current_length += len(part)
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    async def _smooth_audio_streaming(self, websocket):
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            if not self.playback_thread or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._continuous_playback_worker, args=(stream,))
                self.playback_thread.daemon = True
                self.playback_thread.start()
            total_audio_chunks = 0
            total_audio_bytes = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if "audio" in data and data["audio"]:
                        audio_data = base64.b64decode(data["audio"])
                        total_audio_chunks += 1
                        total_audio_bytes += len(audio_data)
                        try:
                            self.continuous_buffer.put_nowait(audio_data)
                        except queue.Full:
                            pass
                        self.mp3_chunks.append(audio_data)
                        self.all_audio_chunks.append(audio_data)
                        if not self.playback_started and total_audio_chunks >= self.BUFFER_SIZE:
                            self.playback_started = True
                            print(f"🎵 Starting continuous playback (buffer: {total_audio_chunks} chunks)")
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    if data.get("isFinal"):
                        print("✅ TTS stream completed")
                        break
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")

    def _continuous_playback_worker(self, stream):
        try:
            while self.running:
                try:
                    audio_data = self.continuous_buffer.get(timeout=0.1)
                    self._play_audio_chunk_smooth(audio_data, stream)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    continue
            print("🎵 Continuous playback worker completed")
        except Exception as e:
            print(f"❌ Continuous playback worker error: {e}")

    def _play_audio_chunk_smooth(self, mp3_data, stream):
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
            stream.write(pcm_float.astype(np.float32).tobytes())
        except Exception as e:
            pass

    def start(self):
        print("🎤🎵 Real-time Voice Changer with Smooth Live Playback (Pro)")
        print("=" * 60)
        print("🎤 STT: Captures your speech → adds to queue")
        print("📝 Text Processor: Processes from memory queue")
        print("🎵 TTS: Streams text → smooth live playback (Pro Multilingual v2 Model)")
        print(f"📁 Queue log: {self.QUEUE_LOG_FILE}")
        print(f"🎵 Output file: {self.OUTPUT_MP3_FILE}")
        print("=" * 60)
        if os.path.exists(self.QUEUE_LOG_FILE):
            os.remove(self.QUEUE_LOG_FILE)
        stt_thread = threading.Thread(target=self.stt_worker)
        stt_thread.daemon = True
        stt_thread.start()
        self.text_processor_thread = threading.Thread(target=self.text_processor_worker)
        self.text_processor_thread.daemon = True
        self.text_processor_thread.start()
        print("✅ Real-time voice changer (Pro) started!")
        print("🎤 Speak into your microphone...")
        print("🎵 Your voice will be changed in real-time!")
        print("⏹️  Press Ctrl+C to stop")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping real-time voice changer...")
            self.running = False
            self.save_final_output()
            print("✅ Real-time voice changer stopped")

    def save_final_output(self):
        try:
            if self.all_audio_chunks:
                print(f"💾 Saving {len(self.all_audio_chunks)} audio chunks to {self.OUTPUT_MP3_FILE}...")
                combined_audio = b''.join(self.all_audio_chunks)
                with open(self.OUTPUT_MP3_FILE, 'wb') as f:
                    f.write(combined_audio)
                print(f"✅ All audio saved to: {self.OUTPUT_MP3_FILE}")
                print(f"📊 Total audio size: {len(combined_audio)} bytes")
                self.log_to_file(f"FINAL: Saved {len(self.all_audio_chunks)} chunks, {len(combined_audio)} bytes")
            else:
                print("⚠️ No audio chunks to save")
                self.log_to_file("FINAL: No audio chunks to save")
        except Exception as e:
            print(f"❌ Error saving final output: {e}")
            self.log_to_file(f"ERROR: Failed to save final output - {e}")

async def main():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Please set the path to your Google Cloud service account key")
        return
    print("✅ Prerequisites check passed")
    voice_changer = RealtimeVoiceChangerSmoothPro()
    voice_changer.start()

if __name__ == "__main__":
    asyncio.run(main()) 