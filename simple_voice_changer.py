#!/usr/bin/env python3
"""
Simple Voice Changer - Single Process
Combines STT and TTS in one process like websocket_streaming_test.py
"""

import os
import asyncio
import json
import base64
import websockets
import pyaudio
import numpy as np
import time
import threading
from google.cloud import speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleVoiceChanger:
    def __init__(self):
        # ElevenLabs settings
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paFloat32
        self.CHUNK_SIZE = 1024
        
        # STT settings
        self.STT_SAMPLE_RATE = 16000
        self.STT_CHUNK_SIZE = 1024 * 4
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Google Cloud Speech client
        self.speech_client = None
        self.init_google_client()
        
        # Control flags
        self.running = True
        self.current_text = ""
        self.text_lock = threading.Lock()
        
        # WebSocket
        self.websocket = None
    
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
                sample_rate_hertz=self.STT_SAMPLE_RATE,
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
    
    async def stream_text_and_play_audio(self, text):
        """Stream text to ElevenLabs and play audio immediately"""
        if not self.websocket:
            return
        
        try:
            print(f"🎵 TTS: Streaming text: '{text}'")
            
            # Send text
            text_message = {
                "text": text,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(text_message))
            
            # Send end marker
            await self.websocket.send(json.dumps({"text": ""}))
            
            # Initialize audio output
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎵 TTS: Audio output stream initialized")
            
            # Receive and play audio
            async for message in self.websocket:
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
            print(f"❌ TTS: Error in stream_text_and_play_audio: {e}")
    
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
    
    async def stt_worker(self):
        """STT worker - continuously transcribe audio"""
        print("🎤 STT Worker: Starting...")
        
        try:
            # Initialize audio input
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.STT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.STT_CHUNK_SIZE
            )
            
            print("✅ STT: Audio input stream started")
            
            # Buffer for accumulating audio chunks
            audio_buffer = b""
            buffer_duration = 2.0  # Process every 2 seconds
            chunk_duration = self.STT_CHUNK_SIZE / self.STT_SAMPLE_RATE
            chunks_per_buffer = int(buffer_duration / chunk_duration)
            
            chunk_count = 0
            
            while self.running:
                try:
                    # Read audio chunk
                    audio_chunk = stream.read(self.STT_CHUNK_SIZE, exception_on_overflow=False)
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
                                
                                # Set current text for TTS
                                with self.text_lock:
                                    self.current_text = transcribed_text
                                    print(f"📚 STT: Set text for TTS: '{transcribed_text}'")
                            else:
                                print("🔇 STT: No speech detected")
                        
                        # Reset buffer
                        audio_buffer = b""
                        chunk_count = 0
                    
                    await asyncio.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            print("✅ STT: Worker stopped")
            
        except Exception as e:
            print(f"❌ STT: Worker error: {e}")
    
    async def tts_worker(self):
        """TTS worker - continuously check for new text and stream it"""
        print("🎵 TTS Worker: Starting...")
        
        try:
            # Connect to WebSocket
            self.websocket = await self.connect_websocket()
            if not self.websocket:
                return
            
            last_processed_text = ""
            
            while self.running:
                try:
                    # Check for new text
                    with self.text_lock:
                        current_text = self.current_text
                    
                    if current_text and current_text != last_processed_text:
                        # Stream new text
                        await self.stream_text_and_play_audio(current_text)
                        last_processed_text = current_text
                        
                        # Clear the text after processing
                        with self.text_lock:
                            self.current_text = ""
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ TTS: Error in TTS worker: {e}")
                    break
            
            # Cleanup
            if self.websocket:
                await self.websocket.close()
            print("✅ TTS: Worker stopped")
            
        except Exception as e:
            print(f"❌ TTS: Worker error: {e}")
    
    async def run(self):
        """Main async function"""
        print("🚀 Simple Voice Changer - Single Process")
        print("=" * 50)
        print("📖 Speak in Russian and hear it transformed in real-time!")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        
        # Run STT and TTS workers concurrently
        await asyncio.gather(
            self.stt_worker(),
            self.tts_worker()
        )

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
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and run voice changer
    voice_changer = SimpleVoiceChanger()
    
    try:
        asyncio.run(voice_changer.run())
    except KeyboardInterrupt:
        print("\n⏹️  Stopping voice changer...")
        voice_changer.running = False
        print("👋 Voice changer stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 