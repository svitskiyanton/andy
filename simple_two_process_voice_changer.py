#!/usr/bin/env python3
"""
Simple Two-Process Voice Changer
Uses 2 processes: STT (Google) and TTS (ElevenLabs) with shared text variable
"""

import os
import multiprocessing
import time
import json
import base64
import websockets
import pyaudio
import numpy as np
from dotenv import load_dotenv
from google.cloud import speech
import asyncio

# Load environment variables
load_dotenv()

class STTProcess:
    """Speech-to-Text process using Google Cloud"""
    
    def __init__(self, shared_text, text_lock, running_flag):
        self.shared_text = shared_text
        self.text_lock = text_lock
        self.running_flag = running_flag
        
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.CHUNK_SIZE = 1024 * 4
        
        # Will be initialized in run()
        self.speech_client = None
        self.audio = None
    
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
            # Create audio content
            audio = speech.RecognitionAudio(content=audio_chunk)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code="ru-RU",  # Russian
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract transcribed text
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
    
    def run(self):
        """Main STT process loop"""
        print("🎤 STT Process: Starting...")
        
        try:
            # Initialize Google Cloud client in this process
            self.init_google_client()
            
            # Initialize PyAudio in this process
            self.audio = pyaudio.PyAudio()
            
            # Initialize audio input
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("✅ STT: Audio input stream started")
            
            # Buffer for accumulating audio chunks
            audio_buffer = b""
            buffer_duration = 2.0  # Process every 2 seconds
            chunk_duration = self.CHUNK_SIZE / self.SAMPLE_RATE
            chunks_per_buffer = int(buffer_duration / chunk_duration)
            
            chunk_count = 0
            
            while self.running_flag.value:
                try:
                    # Read audio chunk
                    audio_chunk = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_buffer += audio_chunk
                    chunk_count += 1
                    
                    # Process buffer when we have enough chunks
                    if chunk_count >= chunks_per_buffer:
                        if audio_buffer:
                            print(f"🎤 STT: Processing {len(audio_buffer)} bytes of audio...")
                            
                            # Transcribe audio
                            transcribed_text = self.transcribe_audio(audio_buffer)
                            
                            if transcribed_text:
                                print(f"📝 STT: Transcribed: '{transcribed_text}'")
                                
                                # Append to shared text
                                with self.text_lock:
                                    current_text = self.shared_text.value
                                    new_text = current_text + transcribed_text + " "
                                    self.shared_text.value = new_text
                                    print(f"📚 STT: Shared text (FULL): '{new_text}'")
                            else:
                                print("🔇 STT: No speech detected")
                        
                        # Reset buffer
                        audio_buffer = b""
                        chunk_count = 0
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            print("✅ STT: Process stopped")
            
        except Exception as e:
            print(f"❌ STT: Process error: {e}")


class TTSProcess:
    """Text-to-Speech process using ElevenLabs WebSocket"""
    
    def __init__(self, shared_text, text_lock, running_flag):
        self.shared_text = shared_text
        self.text_lock = text_lock
        self.running_flag = running_flag
        
        # ElevenLabs settings
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # Audio - will be initialized in run()
        self.audio = None
        
        # Track processed text
        self.last_processed_length = 0
    
    async def connect_websocket(self):
        """Connect to ElevenLabs WebSocket"""
        try:
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            websocket = await websockets.connect(uri, extra_headers=headers)
            print("✅ TTS: Connected to ElevenLabs WebSocket")
            
            # Send initialization
            init_message = {
                "text": " ",
                "voice_settings": {
                    "speed": 1,
                    "stability": 0.5,
                    "similarity_boost": 0.8
                },
                "xi_api_key": self.ELEVENLABS_API_KEY
            }
            await websocket.send(json.dumps(init_message))
            print("📤 TTS: Sent initialization message")
            
            return websocket
            
        except Exception as e:
            print(f"❌ TTS: Failed to connect to ElevenLabs WebSocket: {e}")
            return None
    
    async def send_text_and_receive_audio(self, websocket, text):
        """Send text to ElevenLabs and receive audio"""
        try:
            print(f"🎵 TTS: Streaming text: '{text}'")
            
            # Send text
            text_message = {
                "text": text,
                "try_trigger_generation": True
            }
            await websocket.send(json.dumps(text_message))
            
            # Send end marker
            await websocket.send(json.dumps({"text": ""}))
            
            # Initialize audio output
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                frames_per_buffer=1024
            )
            
            print("🎵 TTS: Audio output stream initialized")
            
            # Receive and play audio
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Decode and play audio
                        audio_data = base64.b64decode(data["audio"])
                        await self._play_audio_chunk(audio_data, stream)
                        print(f"🔊 TTS: Audio chunk: {len(audio_data)} bytes")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 TTS: End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ TTS: Stream completed")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ TTS: JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ TTS: Message processing error: {e}")
                    break
            
            # Cleanup audio stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ TTS: Error in send_text_and_receive_audio: {e}")
    
    async def _play_audio_chunk(self, mp3_data, stream):
        """Play a single audio chunk"""
        try:
            from pydub import AudioSegment
            import io
            
            # Validate MP3 data
            if len(mp3_data) < 100:
                return
            
            # Decode MP3 to PCM
            audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            
            if len(audio_segment) == 0:
                return
            
            # Convert to PCM
            pcm_data = audio_segment.get_array_of_samples()
            pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
            
            # Play in chunks
            chunk_size = 1024
            for i in range(0, len(pcm_float), chunk_size):
                chunk = pcm_float[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                stream.write(chunk.astype(np.float32).tobytes())
                
        except Exception as e:
            print(f"⚠️ TTS: Audio chunk playback error: {e}")
    
    async def run_async(self):
        """Main TTS async process loop"""
        print("🎵 TTS Process: Starting...")
        
        try:
            while self.running_flag.value:
                try:
                    # Connect to WebSocket
                    websocket = await self.connect_websocket()
                    if not websocket:
                        await asyncio.sleep(1)
                        continue
                    
                    while self.running_flag.value:
                        # Check if there's new text to process
                        with self.text_lock:
                            current_text = self.shared_text.value
                        
                        if len(current_text) > self.last_processed_length:
                            # Get new text to stream
                            new_text = current_text[self.last_processed_length:].strip()
                            
                            if new_text:
                                # Send text and receive audio
                                try:
                                    await self.send_text_and_receive_audio(websocket, new_text)
                                    self.last_processed_length = len(current_text)
                                    # Clear the shared text buffer after playing
                                    with self.text_lock:
                                        self.shared_text.value = ""
                                    self.last_processed_length = 0
                                except Exception as e:
                                    print(f"❌ TTS: Error in send_text_and_receive_audio: {e}")
                                    break  # Break to reconnect WebSocket
                        
                        await asyncio.sleep(0.1)
                    
                    # Cleanup
                    await websocket.close()
                    self.audio.terminate()
                    print("✅ TTS: Process stopped")
                except Exception as e:
                    print(f"❌ TTS: Process error: {e}")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ TTS: Fatal process error: {e}")
    
    def run(self):
        """Main TTS process entry point"""
        # Initialize PyAudio in this process
        self.audio = pyaudio.PyAudio()
        asyncio.run(self.run_async())


def main():
    """Main entry point"""
    print("🚀 Simple Two-Process Voice Changer")
    print("=" * 50)
    
    # Check required environment variables
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        return
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Please set the path to your Google Cloud service account key")
        return
    
    print("✅ Prerequisites check passed")
    print("📖 Speak in Russian and hear it transformed in real-time!")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    # Create shared variables
    manager = multiprocessing.Manager()
    shared_text = manager.Value('c', "")  # Shared string
    text_lock = manager.Lock()
    running_flag = manager.Value('b', True)
    
    try:
        # Create processes
        stt_process = STTProcess(shared_text, text_lock, running_flag)
        tts_process = TTSProcess(shared_text, text_lock, running_flag)
        
        # Start processes
        stt_proc = multiprocessing.Process(target=stt_process.run)
        tts_proc = multiprocessing.Process(target=tts_process.run)
        
        stt_proc.start()
        tts_proc.start()
        
        print("✅ Both processes started")
        
        # Wait for processes
        stt_proc.join()
        tts_proc.join()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping voice changer...")
        running_flag.value = False
        
        # Wait for processes to finish
        if stt_proc.is_alive():
            stt_proc.terminate()
            stt_proc.join()
        if tts_proc.is_alive():
            tts_proc.terminate()
            tts_proc.join()
        
        print("👋 Voice changer stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        running_flag.value = False


if __name__ == "__main__":
    main() 