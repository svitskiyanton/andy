#!/usr/bin/env python3
"""
Simple Streaming Voice Changer
Combines STT (Google) and TTS (ElevenLabs WebSocket) using shared text buffer
"""

import os
import asyncio
import json
import base64
import websockets
import pyaudio
import numpy as np
from dotenv import load_dotenv
import time
import threading
import queue
from google.cloud import speech

# Load environment variables
load_dotenv()

class SimpleStreamingVoiceChanger:
    def __init__(self):
        # ElevenLabs settings
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # Google Cloud settings
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Audio settings
        self.SAMPLE_RATE = 16000  # Google STT requires 16kHz
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.CHUNK_SIZE = 1024 * 4
        
        # Shared text buffer
        self.shared_text = ""
        self.text_lock = threading.Lock()
        
        # Audio queues
        self.audio_queue = queue.Queue()
        
        # Control flags
        self.running = False
        self.audio = pyaudio.PyAudio()
        
        # Initialize Google Speech client
        self.speech_client = None
        self.init_google_client()
        
        # WebSocket connection
        self.websocket = None
        
        print("🎤 Simple Streaming Voice Changer initialized")
        print(f"🎵 Voice ID: {self.VOICE_ID}")
        print(f"🌍 Model: {self.MODEL_ID}")
    
    def init_google_client(self):
        """Initialize Google Cloud Speech client"""
        try:
            if self.GOOGLE_APPLICATION_CREDENTIALS:
                self.speech_client = speech.SpeechClient()
                print("✅ Google Cloud Speech client initialized")
            else:
                print("⚠️  Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud client: {e}")
    
    async def connect_elevenlabs_websocket(self):
        """Connect to ElevenLabs WebSocket for streaming TTS"""
        try:
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            # Create single connection
            self.websocket = await websockets.connect(uri, extra_headers=headers)
            print("✅ Connected to ElevenLabs WebSocket")
            
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
            await self.websocket.send(json.dumps(init_message))
            print("📤 Sent initialization message")
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect to ElevenLabs WebSocket: {e}")
            return False
    
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
            
            # Extract transcribed text (same as working test script)
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            text = transcribed_text.strip()
            if text:
                return text
            else:
                return None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for input audio"""
        if self.running:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def output_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for output audio"""
        if self.running and not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get_nowait()
                return (audio_data, pyaudio.paContinue)
            except queue.Empty:
                pass
        return (b'\x00' * frame_count * 2, pyaudio.paContinue)
    
    async def stt_worker(self):
        """STT worker that processes audio and appends to shared text"""
        print("🎤 Starting STT worker...")
        
        # Buffer for accumulating audio chunks
        audio_buffer = b""
        buffer_duration = 2.0  # Process every 2 seconds
        chunk_duration = self.CHUNK_SIZE / self.SAMPLE_RATE
        chunks_per_buffer = int(buffer_duration / chunk_duration)
        
        chunk_count = 0
        
        while self.running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer += audio_chunk
                chunk_count += 1
                
                # Process buffer when we have enough chunks
                if chunk_count >= chunks_per_buffer:
                    if audio_buffer:
                        print(f"🎤 Processing {len(audio_buffer)} bytes of audio...")
                        
                        # Transcribe audio
                        transcribed_text = self.transcribe_audio(audio_buffer)
                        
                        if transcribed_text:
                            print(f"📝 Transcribed: '{transcribed_text}'")
                            
                            # Append to shared text buffer
                            with self.text_lock:
                                self.shared_text += transcribed_text + " "
                                print(f"📚 Shared text buffer (FULL): '{self.shared_text}'")
                        else:
                            print("🔇 No speech detected")
                    
                    # Reset buffer
                    audio_buffer = b""
                    chunk_count = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in STT worker: {e}")
                break
    
    async def tts_worker(self):
        """TTS worker that streams text from shared buffer and receives audio"""
        print("🎵 Starting TTS worker...")
        
        last_processed_length = 0
        
        # Initialize audio stream
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            output=True,
            frames_per_buffer=1024
        )
        
        print("🎵 Audio output stream initialized")
        
        while self.running:
            try:
                # Check if there's new text to stream
                with self.text_lock:
                    current_text = self.shared_text
                
                if len(current_text) > last_processed_length:
                    # Get new text to stream
                    new_text = current_text[last_processed_length:].strip()
                    
                    if new_text:
                        print(f"🎵 Streaming new text: '{new_text}'")
                        
                        # Send text to ElevenLabs
                        text_message = {
                            "text": new_text,
                            "try_trigger_generation": True
                        }
                        await self.websocket.send(json.dumps(text_message))
                        
                        # Send end marker
                        await self.websocket.send(json.dumps({"text": ""}))
                        
                        # Receive and play audio response
                        async for message in self.websocket:
                            try:
                                data = json.loads(message)
                                
                                if "audio" in data and data["audio"]:
                                    # Decode and play audio
                                    audio_data = base64.b64decode(data["audio"])
                                    await self._play_audio_chunk(audio_data, stream)
                                    print(f"🔊 Audio chunk: {len(audio_data)} bytes")
                                
                                elif "audio" in data and data["audio"] is None:
                                    print("📡 End of stream signal received")
                                
                                if data.get("isFinal"):
                                    print("✅ Stream completed")
                                    break
                                    
                            except json.JSONDecodeError as e:
                                print(f"⚠️ JSON decode error: {e}")
                                continue
                            except Exception as e:
                                print(f"❌ Message processing error: {e}")
                                break
                        
                        last_processed_length = len(current_text)
                
                # Small delay to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error in TTS worker: {e}")
                break
        
        # Cleanup
        stream.stop_stream()
        stream.close()
    
    async def audio_receiver(self):
        """Receive and play audio from ElevenLabs WebSocket"""
        print("🔊 Starting audio receiver...")
        
        try:
            # Initialize audio stream
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                frames_per_buffer=1024
            )
            
            print("🎵 Audio output stream initialized")
            
            # Receive and play audio chunks
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Decode and play audio
                        audio_data = base64.b64decode(data["audio"])
                        await self._play_audio_chunk(audio_data, stream)
                        print(f"🔊 Audio chunk: {len(audio_data)} bytes")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ Stream completed")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Audio receiver error: {e}")
    
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
            print(f"⚠️ Audio chunk playback error: {e}")
    
    def start_audio_streams(self):
        """Start PyAudio input and output streams"""
        try:
            # Input stream (microphone)
            self.input_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self.audio_callback
            )
            
            self.input_stream.start_stream()
            print("✅ Audio input stream started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start audio streams: {e}")
            return False
    
    def stop_audio_streams(self):
        """Stop PyAudio streams"""
        try:
            if hasattr(self, 'input_stream'):
                self.input_stream.stop_stream()
                self.input_stream.close()
            
            self.audio.terminate()
            print("✅ Audio streams stopped")
            
        except Exception as e:
            print(f"❌ Error stopping audio streams: {e}")
    
    async def run(self):
        """Main run method"""
        print("🚀 Starting simple streaming voice changer...")
        print("📖 Speak in Russian and hear it transformed in real-time!")
        print("⏹️  Press Ctrl+C to stop")
        
        # Connect to ElevenLabs WebSocket
        if not await self.connect_elevenlabs_websocket():
            return
        
        # Start audio streams
        if not self.start_audio_streams():
            return
        
        self.running = True
        
        try:
            # Start all workers concurrently
            await asyncio.gather(
                self.stt_worker(),
                self.tts_worker()
            )
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping voice changer...")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.running = False
            self.stop_audio_streams()
            
            if self.websocket:
                await self.websocket.close()
            
            print("👋 Voice changer stopped")

def main():
    """Main entry point"""
    # Check required environment variables
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("❌ ELEVENLABS_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        return
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Please set the path to your Google Cloud service account key")
        print("   Get it from: https://console.cloud.google.com/apis/credentials")
        return
    
    # Create and run voice changer
    voice_changer = SimpleStreamingVoiceChanger()
    
    try:
        asyncio.run(voice_changer.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 