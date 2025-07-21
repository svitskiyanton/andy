#!/usr/bin/env python3
"""
Combined Speechmatics + ElevenLabs Real-time Voice Changer
Properly combines the working patterns from both implementations
"""

import asyncio
import base64
import json
import os
import threading
import time
import uuid
import websockets
import pyaudio
import queue
from dotenv import load_dotenv
from typing import AsyncGenerator

# Import Speechmatics models for better type safety and validation
try:
    from speechmatics.models import (
        AudioSettings,
        TranscriptionConfig,
    )
    SPEECHMATICS_MODELS_AVAILABLE = True
except ImportError:
    SPEECHMATICS_MODELS_AVAILABLE = False
    print("⚠️  Speechmatics models not available, using manual config")

# Load environment variables
load_dotenv()

# Configuration
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
SM_API_KEY = os.getenv("SM_API_KEY")

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Speechmatics configuration
CONNECTION_URL = "wss://eu2.rt.speechmatics.com/v2"

class SpeechmaticsSTT:
    """Speechmatics STT following the working pattern"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or SM_API_KEY
        self.websocket = None
        self.connected = False
        self.transcript_queue = queue.Queue()
        self.audio_chunks_sent = 0
        
        if not self.api_key:
            raise ValueError("SM_API_KEY not found in environment variables")
    
    async def connect(self):
        """Connect to Speechmatics STT"""
        if self.connected:
            return
            
        print("🔗 Connecting to Speechmatics STT...")
        
        try:
            self.websocket = await websockets.connect(
                CONNECTION_URL,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
                ping_interval=30,
                ping_timeout=60,
                close_timeout=5
            )
            print("✅ Connected to Speechmatics STT")
            
            # Send configuration following working pattern with all parameters
            if SPEECHMATICS_MODELS_AVAILABLE:
                # Use Speechmatics models for proper validation
                audio_settings = AudioSettings(
                    encoding="pcm_s16le",
                    sample_rate=16000,
                    chunk_size=CHUNK_SIZE * 2  # 2048 samples like working version
                )
                
                transcription_config = TranscriptionConfig(
                    language="ru",                    # Russian language
                    max_delay=1.0,                    # Ultra-low latency
                    max_delay_mode="flexible",        # Better entity formatting
                    operating_point="enhanced",       # Highest accuracy model
                    enable_partials=True              # Critical for <500ms perceived latency
                )
                
                config = {
                    "message": "StartRecognition",
                    "audio_format": {
                        "type": "raw",  # Add type manually since AudioSettings doesn't have it
                        "encoding": audio_settings.encoding,
                        "sample_rate": audio_settings.sample_rate
                    },
                    "transcription_config": transcription_config.asdict()
                }
            else:
                # Manual configuration fallback
                config = {
                    "message": "StartRecognition",
                    "audio_format": {
                        "type": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": 16000
                    },
                    "transcription_config": {
                        "language": "ru",                    # Russian language
                        "max_delay": 1.0,                    # Ultra-low latency
                        "max_delay_mode": "flexible",        # Better entity formatting
                        "operating_point": "enhanced",       # Highest accuracy model
                        "enable_partials": True              # Critical for <500ms perceived latency
                    }
                }
            
            await self.websocket.send(json.dumps(config))
            self.connected = True
            print("🚀 STT ready for transcription!")
            
        except Exception as e:
            print(f"❌ STT connection failed: {e}")
            self.connected = False
            raise
    
    async def receive_handler(self):
        """Handle incoming messages from Speechmatics"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("message")
                    
                    if message_type == "RecognitionStarted":
                        session_id = data.get("id", "unknown")
                        print(f"📝 Recognition started: {session_id}")
                        
                    elif message_type == "AddTranscript":
                        # Final transcript
                        transcript = data.get("metadata", {}).get("transcript", "")
                        if transcript:
                            print(f"📝 Final: '{transcript}'")
                            self.transcript_queue.put(("final", transcript))
                            
                    elif message_type == "AddPartialTranscript":
                        # Partial transcript
                        transcript = data.get("metadata", {}).get("transcript", "")
                        if transcript:
                            print(f"📝 Partial: '{transcript}'")
                            self.transcript_queue.put(("partial", transcript))
                            
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            print(f"❌ STT receive error: {e}")
    
    async def send_audio(self, audio_data):
        """Send audio data to Speechmatics"""
        if not self.connected:
            return
        
        try:
            await self.websocket.send(audio_data)
            self.audio_chunks_sent += 1
        except Exception as e:
            print(f"❌ Failed to send audio: {e}")
    
    def get_transcript(self):
        """Get latest transcript from queue"""
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None
    
    async def close(self):
        """Close STT connection"""
        if self.connected and self.websocket:
            try:
                # Send EndOfStream message
                end_message = {
                    "message": "EndOfStream",
                    "last_seq_no": self.audio_chunks_sent
                }
                await self.websocket.send(json.dumps(end_message))
                
                await self.websocket.close()
                print("🔌 STT connection closed")
            except Exception as e:
                print(f"❌ Error closing STT connection: {e}")
            finally:
                self.connected = False

class ElevenLabsTTS:
    """ElevenLabs TTS following the working pattern"""
    
    def __init__(self, voice_id=None, api_key=None):
        self.voice_id = voice_id or EL_VOICE_ID
        self.api_key = api_key or EL_API_KEY
        self.websocket = None
        self.connected = False
        self.context_id = None
        self.audio_queue = queue.Queue()
        self.message_queue = queue.Queue()
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
    
    async def connect(self):
        """Connect to ElevenLabs TTS"""
        if self.connected:
            return
            
        print("🔗 Connecting to ElevenLabs TTS...")
        
        # Connect to multi-stream endpoint
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
        params = {
            "model_id": "eleven_flash_v2_5",
            "output_format": "pcm_16000",
            "auto_mode": "true"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"
        
        try:
            self.websocket = await websockets.connect(
                full_url,
                extra_headers={"xi-api-key": self.api_key},
                close_timeout=5
            )
            print("✅ Connected to ElevenLabs TTS")
            
            # Initialize context
            self.context_id = f"conv_{uuid.uuid4().hex[:8]}"
            context_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                },
                "context_id": self.context_id
            }
            
            await self.websocket.send(json.dumps(context_message))
            print(f"✅ TTS context initialized: {self.context_id}")
            
            # Start message receiver
            asyncio.create_task(self._message_receiver())
            
            self.connected = True
            print("🚀 TTS ready for speech synthesis!")
            
        except Exception as e:
            print(f"❌ TTS connection failed: {e}")
            self.connected = False
            raise
    
    async def _message_receiver(self):
        """Receive messages from ElevenLabs"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.message_queue.put(data)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"❌ TTS message receiver error: {e}")
    
    async def speak(self, text):
        """Convert text to speech"""
        if not self.connected:
            return None
        
        print(f"🗣️  Speaking: '{text}'")
        
        try:
            # Send text
            message = {
                "text": text + " ",
                "context_id": self.context_id,
                "flush": True
            }
            
            await self.websocket.send(json.dumps(message))
            
            # Receive audio with better timeout handling
            audio_chunks = []
            max_chunks = 15
            timeout = 5  # Reduced timeout
            start_time = time.time()
            
            while len(audio_chunks) < max_chunks and (time.time() - start_time) < timeout:
                try:
                    data = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                    
                    if "audio" in data:
                        audio_bytes = base64.b64decode(data["audio"])
                        audio_chunks.append(audio_bytes)
                        print(f"🎵 Received audio chunk: {len(audio_bytes)} bytes")
                        
                        if data.get("is_final", False):
                            print("✅ Audio generation complete")
                            break
                            
                except asyncio.TimeoutError:
                    print("⏰ TTS timeout, continuing...")
                    break
                except Exception as e:
                    print(f"❌ Audio processing error: {e}")
                    break
            
            if audio_chunks:
                audio_data = b''.join(audio_chunks)
                self.audio_queue.put(audio_data)
                print(f"🎵 Generated {len(audio_data)} bytes of audio")
                return audio_data
            else:
                print("⚠️  No audio generated")
            
            return None
            
        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return None
    
    def get_audio(self):
        """Get latest audio from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    async def close(self):
        """Close TTS connection"""
        if self.connected and self.websocket:
            try:
                # Close context
                close_message = {
                    "context_id": self.context_id,
                    "close_context": True
                }
                await self.websocket.send(json.dumps(close_message))
                
                # Close socket
                close_socket_message = {
                    "close_socket": True
                }
                await self.websocket.send(json.dumps(close_socket_message))
                
                await self.websocket.close()
                print("🔌 TTS connection closed")
                
            except Exception as e:
                print(f"❌ Error closing TTS connection: {e}")
            finally:
                self.connected = False

class CombinedVoiceChanger:
    """Combined real-time voice changer"""
    
    def __init__(self):
        self.stt = SpeechmaticsSTT()
        self.tts = ElevenLabsTTS()
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.running = False
        
        # Audio processing
        self.audio_buffer = b''
        self.buffer_size = RATE * 2  # 2 seconds of audio
        self.last_transcript = ""
        
    def start_audio_streams(self):
        """Start microphone input and speaker output streams"""
        # Input stream (microphone)
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Output stream (speakers)
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("🎤 Audio streams started")
    
    async def audio_input_loop(self):
        """Continuously capture audio from microphone"""
        print("🎤 Starting audio input loop...")
        
        while self.running:
            try:
                # Read audio chunk
                audio_chunk = self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.audio_buffer += audio_chunk
                
                # Send to STT when buffer is full
                if len(self.audio_buffer) >= self.buffer_size:
                    await self.stt.send_audio(self.audio_buffer)
                    self.audio_buffer = b''  # Reset buffer
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                print(f"❌ Audio input error: {e}")
                break
    
    async def audio_output_loop(self):
        """Continuously play audio from TTS"""
        print("🔊 Starting audio output loop...")
        
        while self.running:
            try:
                # Get audio from TTS queue
                audio_data = self.tts.get_audio()
                if audio_data:
                    print(f"🎵 Playing {len(audio_data)} bytes of audio")
                    # Play audio directly
                    self.output_stream.write(audio_data)
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                print(f"❌ Audio output error: {e}")
                break
    
    async def transcription_loop(self):
        """Process transcriptions and generate speech"""
        print("📝 Starting transcription loop...")
        
        while self.running:
            try:
                # Get transcript from STT
                transcript_data = self.stt.get_transcript()
                if transcript_data:
                    transcript_type, transcript = transcript_data
                    
                    # Only process final transcripts to avoid spam
                    if transcript_type == "final" and transcript != self.last_transcript:
                        self.last_transcript = transcript
                        await self.tts.speak(transcript)
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"❌ Transcription processing error: {e}")
                break
    
    async def start(self):
        """Start the combined voice changer"""
        print("🚀 Starting combined voice changer...")
        
        try:
            # Connect to services
            await asyncio.gather(
                self.stt.connect(),
                self.tts.connect()
            )
            
            # Start audio streams
            self.start_audio_streams()
            
            # Start STT receiver
            stt_receiver = asyncio.create_task(self.stt.receive_handler())
            
            # Start processing loops
            self.running = True
            
            await asyncio.gather(
                self.audio_input_loop(),
                self.audio_output_loop(),
                self.transcription_loop()
            )
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping voice changer...")
        except Exception as e:
            print(f"❌ Voice changer error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the voice changer"""
        self.running = False
        
        # Close audio streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()
        
        # Close service connections
        await asyncio.gather(
            self.stt.close(),
            self.tts.close()
        )
        
        print("✅ Voice changer stopped")

async def main():
    """Main function"""
    voice_changer = CombinedVoiceChanger()
    
    try:
        await voice_changer.start()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main()) 