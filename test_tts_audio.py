#!/usr/bin/env python3
"""
Simple TTS audio test to verify audio output quality
"""

import asyncio
import base64
import json
import os
import threading
import queue
import uuid
import websockets
import pyaudio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

if not EL_API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found in .env file")
    exit(1)

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # ElevenLabs uses 24kHz

class SimpleAudioPlayer:
    """Simple audio player for real-time playback"""
    
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
    
    def run(self):
        try:
            print(f"🔊 Initializing audio player: {RATE}Hz, {CHANNELS} channel(s), {FORMAT}")
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print("🔊 Audio player initialized successfully")
            
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk:
                        print(f"🔊 Playing audio chunk: {len(chunk)} bytes")
                        self.stream.write(chunk)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Audio player error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Failed to initialize audio player: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
    
    def stop(self):
        self.running = False

class SimpleElevenLabsTTS:
    """Simple ElevenLabs TTS for testing"""
    
    def __init__(self):
        self.voice_id = EL_VOICE_ID
        self.api_key = EL_API_KEY
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.player = None
        self.player_thread = None
        
    def start_audio_player(self):
        """Start the audio player thread"""
        self.player = SimpleAudioPlayer(self.audio_queue)
        self.player_thread = threading.Thread(target=self.player.run, daemon=True)
        self.player_thread.start()
        print("🔊 Audio player thread started")
    
    async def connect(self):
        """Establish WebSocket connection to ElevenLabs"""
        print("🔊 Connecting to ElevenLabs...")
        
        try:
            url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
            self.websocket = await websockets.connect(
                url,
                extra_headers={"xi-api-key": self.api_key}
            )
            print("🔊 WebSocket connection established")
            
            # Initialize context
            init_message = {
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                "xi_api_key": self.api_key,
                "generation_config": {"chunk_length_schedule": [50, 90, 120, 150, 200]},
                "model_id": "eleven_multilingual_v2"
            }
            
            await self.websocket.send(json.dumps(init_message))
            print("🔊 Context initialized")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    async def speak(self, text):
        """Send text for TTS"""
        print(f"🔊 Speaking: {text}")
        
        # Send text message
        text_message = {"text": text, "xi_api_key": self.api_key}
        await self.websocket.send(json.dumps(text_message))
        
        # Receive audio
        first_audio = False
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data.get("audio"):
                    audio_data = base64.b64decode(data["audio"])
                    
                    if not first_audio:
                        print(f"🔊 First audio chunk: {len(audio_data)} bytes")
                        first_audio = True
                    
                    self.audio_queue.put(audio_data)
                
                elif data.get("isFinal"):
                    print("🔊 Audio stream completed")
                    break
                    
        except Exception as e:
            print(f"❌ Error receiving audio: {e}")
    
    async def close(self):
        """Close the connection"""
        if self.websocket:
            await self.websocket.close()
        
        if self.player:
            self.player.stop()
            if self.player_thread:
                self.player_thread.join(timeout=1)

async def main():
    """Main test function"""
    print("🔊 Starting TTS audio test...")
    
    tts = SimpleElevenLabsTTS()
    
    try:
        tts.start_audio_player()
        await tts.connect()
        
        # Wait for audio player to initialize
        await asyncio.sleep(1)
        
        # Test with a simple phrase
        await tts.speak("Привет, это тест аудио системы.")
        
        # Wait for audio to finish
        await asyncio.sleep(3)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tts.close()

if __name__ == "__main__":
    asyncio.run(main()) 