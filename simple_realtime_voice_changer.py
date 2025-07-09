#!/usr/bin/env python3
"""
Simple Real-time Voice Changer
STT → File → TTS with smooth playback
"""

import os
import pyaudio
import time
import numpy as np
import threading
import queue
import asyncio
import json
import base64
import websockets
from google.cloud import speech
from dotenv import load_dotenv
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

class SimpleRealtimeVoiceChanger:
    def __init__(self):
        # STT settings
        self.STT_SAMPLE_RATE = 16000
        self.STT_CHANNELS = 1
        self.STT_FORMAT = pyaudio.paInt16
        self.STT_CHUNK_SIZE = 1024
        
        # TTS settings
        self.TTS_SAMPLE_RATE = 44100
        self.TTS_CHANNELS = 1
        self.TTS_FORMAT = pyaudio.paFloat32
        self.TTS_CHUNK_SIZE = 1024
        
        # Audio objects
        self.stt_audio = pyaudio.PyAudio()
        self.tts_audio = pyaudio.PyAudio()
        self.speech_client = None
        
        # File settings
        self.TEXT_FILE = "voice_changer_text.txt"
        
        # Control
        self.running = True
        
        # STT buffers
        self.audio_buffer = []
        self.silence_buffer = []
        self.current_phrase = ""
        self.silence_start = None
        self.max_silence_duration = 2.0
        
        # STT settings
        self.BUFFER_DURATION = 3.0
        self.BUFFER_SIZE_STT = int(self.STT_SAMPLE_RATE * self.BUFFER_DURATION)
        self.SILENCE_DURATION = 1.0
        self.SILENCE_THRESHOLD = 0.01
        
        # TTS playback
        self.audio_queue = queue.Queue(maxsize=50)
        self.playback_thread = None
        
        # File monitoring
        self.last_file_size = 0
        self.last_line_count = 0
        
        # Initialize
        self.init_google_client()
    
    def init_google_client(self):
        """Initialize Google Cloud Speech client"""
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
        """Transcribe audio chunk using Google Speech-to-Text"""
        if not self.speech_client:
            return None
        
        try:
            audio = speech.RecognitionAudio(content=audio_chunk)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.STT_SAMPLE_RATE,
                language_code="ru-RU",
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            text = transcribed_text.strip()
            if text:
                return text
            else:
                return None
            
        except Exception as e:
            print(f"❌ STT: Transcription error: {e}")
            return None
    
    def write_phrase_to_file(self, phrase):
        """Append complete phrase to file"""
        try:
            with open(self.TEXT_FILE, 'a', encoding='utf-8') as f:
                f.write(phrase + "\n")
            print(f"📝 STT → File: '{phrase}'")
        except Exception as e:
            print(f"❌ Error writing to file: {e}")
    
    def stt_worker(self):
        """STT worker thread - captures speech and writes to file"""
        print("🎤 STT Worker: Starting speech capture...")
        
        try:
            stream = self.stt_audio.open(
                format=self.STT_FORMAT,
                channels=self.STT_CHANNELS,
                rate=self.STT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.STT_CHUNK_SIZE
            )
            
            print("✅ STT: Audio input stream started")
            
            while self.running:
                try:
                    data = stream.read(self.STT_CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    self.audio_buffer.extend(audio_chunk)
                    self.silence_buffer.extend(audio_chunk)
                    
                    if len(self.silence_buffer) >= int(self.STT_SAMPLE_RATE * self.SILENCE_DURATION):
                        recent_audio = list(self.silence_buffer)[-int(self.STT_SAMPLE_RATE * self.SILENCE_DURATION):]
                        audio_level = np.sqrt(np.mean(np.array(recent_audio, dtype=np.float32) ** 2))
                    
                        if audio_level < self.SILENCE_THRESHOLD * 32768 and len(self.audio_buffer) >= self.BUFFER_SIZE_STT:
                            stt_chunk = list(self.audio_buffer)
                            self.audio_buffer.clear()
                            self.silence_buffer.clear()
                            
                            stt_audio = np.array(stt_chunk, dtype=np.int16).tobytes()
                            
                            transcribed_text = self.transcribe_audio(stt_audio)
                            
                            if transcribed_text:
                                if self.current_phrase:
                                    self.current_phrase += " " + transcribed_text
                                else:
                                    self.current_phrase = transcribed_text
                                
                                self.silence_start = None
                                
                            else:
                                if self.silence_start is None:
                                    self.silence_start = time.time()
                                else:
                                    silence_duration = time.time() - self.silence_start
                                    if silence_duration >= self.max_silence_duration and self.current_phrase:
                                        self.write_phrase_to_file(self.current_phrase)
                                        self.current_phrase = ""
                                        self.silence_start = None
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            print("✅ STT Worker stopped")
            
        except Exception as e:
            print(f"❌ STT Worker error: {e}")
    
    def file_monitor_worker(self):
        """Monitor text file for new content and trigger TTS"""
        print("📁 File Monitor: Watching for new text...")
        
        while self.running:
            try:
                if os.path.exists(self.TEXT_FILE):
                    current_size = os.path.getsize(self.TEXT_FILE)
                    
                    if current_size > self.last_file_size:
                        print(f"📁 File Monitor: New content! Size: {self.last_file_size} → {current_size}")
                        
                        with open(self.TEXT_FILE, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        print(f"📁 File Monitor: Found {len(lines)} lines, processing from {self.last_line_count}")
                        
                        new_lines = lines[self.last_line_count:]
                        for line in new_lines:
                            if line.strip():
                                text = line.strip()
                                print(f"🎵 TTS: Processing: '{text}'")
                                
                                # Start TTS streaming
                                threading.Thread(target=self.run_tts_streaming, args=(text,)).start()
                        
                        self.last_line_count = len(lines)
                        self.last_file_size = current_size
                    else:
                        print(f"📁 File Monitor: Monitoring... (size: {current_size})")
                else:
                    print("📁 File Monitor: Waiting for file...")
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"❌ File monitor error: {e}")
                time.sleep(1)
    
    def get_api_key(self):
        """Get API key from environment, file, or user input"""
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
    
    def run_tts_streaming(self, text):
        """Run TTS streaming in a separate thread"""
        asyncio.run(self.stream_text_to_tts(text))
    
    async def stream_text_to_tts(self, text):
        """Stream text to TTS with smooth playback"""
        if not text.strip():
            return
        
        # Split text into chunks
        text_chunks = self.split_text_into_chunks(text, max_chunk_size=100)
        
        # Connect to ElevenLabs
        uri = "wss://api.elevenlabs.io/v1/text-to-speech/GN4wbsbejSnGSa1AzjH5/stream-input?model_id=eleven_multilingual_v2&optimize_streaming_latency=4"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Get API key
                api_key = self.get_api_key()
                if not api_key:
                    print("❌ API key not found. Skipping TTS.")
                    return
                
                # Send initialization
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    },
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(init_message))
                
                # Send text chunks
                for chunk in text_chunks:
                    message = {
                        "text": chunk,
                        "xi_api_key": api_key
                    }
                    await websocket.send(json.dumps(message))
                
                # Send end marker
                end_message = {
                    "text": "",
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(end_message))
                
                # Start audio streaming
                await self.stream_audio_from_websocket(websocket)
                
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")
    
    def split_text_into_chunks(self, text, max_chunk_size=100):
        """Split text into chunks while preserving spacing"""
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
    
    async def stream_audio_from_websocket(self, websocket):
        """Stream audio from websocket to playback queue"""
        try:
            # Start playback thread if not already started
            if not self.playback_thread or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self.playback_worker)
                self.playback_thread.daemon = True
                self.playback_thread.start()
                print("🎵 Playback thread started")
            
            # Receive and process audio chunks
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Decode audio
                        audio_data = base64.b64decode(data["audio"])
                        
                        # Add to playback queue
                        try:
                            self.audio_queue.put_nowait(audio_data)
                        except queue.Full:
                            # Queue full, skip this chunk
                            pass
                    
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
            print(f"❌ Audio streaming error: {e}")
    
    def playback_worker(self):
        """Playback worker thread"""
        print("🎵 Playback Worker: Starting...")
        
        try:
            stream = self.tts_audio.open(
                format=self.TTS_FORMAT,
                channels=self.TTS_CHANNELS,
                rate=self.TTS_SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.TTS_CHUNK_SIZE
            )
            
            while self.running:
                try:
                    # Get audio chunk (blocking with timeout)
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # Decode and play
                    self.play_audio_chunk(audio_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue waiting
                    continue
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            print("🎵 Playback worker completed")
            
        except Exception as e:
            print(f"❌ Playback worker error: {e}")
    
    def play_audio_chunk(self, mp3_data, stream):
        """Play audio chunk"""
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
            
            # Play immediately
            stream.write(pcm_float.astype(np.float32).tobytes())
            
        except Exception as e:
            # Silent error handling
            pass
    
    def start(self):
        """Start the real-time voice changer"""
        print("🎤🎵 Simple Real-time Voice Changer")
        print("=" * 50)
        print("🎤 STT: Captures speech → writes to file")
        print("📁 File Monitor: Watches for new text")
        print("🎵 TTS: Streams text → smooth playback")
        print("=" * 50)
        
        # Clear text file
        if os.path.exists(self.TEXT_FILE):
            os.remove(self.TEXT_FILE)
        
        # Start STT worker thread
        stt_thread = threading.Thread(target=self.stt_worker)
        stt_thread.daemon = True
        stt_thread.start()
        
        # Start file monitor thread
        monitor_thread = threading.Thread(target=self.file_monitor_worker)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("✅ Real-time voice changer started!")
        print("🎤 Speak into your microphone...")
        print("🎵 Your voice will be changed in real-time!")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping real-time voice changer...")
            self.running = False
            print("✅ Real-time voice changer stopped")

def main():
    """Main function"""
    # Check prerequisites
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Please set the path to your Google Cloud service account key")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and start voice changer
    voice_changer = SimpleRealtimeVoiceChanger()
    voice_changer.start()

if __name__ == "__main__":
    main() 