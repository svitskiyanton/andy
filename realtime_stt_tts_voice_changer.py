#!/usr/bin/env python3
"""
Real-time Voice Changer using STT→TTS Pipeline
Uses Google Speech-to-Text for Russian language support
and ElevenLabs Text-to-Speech with WebSocket streaming for low latency
"""

import os
import asyncio
import json
import wave
import threading
import time
import queue
import websockets
import pyaudio
import numpy as np
from dotenv import load_dotenv
import requests
from google.cloud import speech
from google.cloud import texttospeech
import io
import base64

# Load environment variables
load_dotenv()

class RealTimeSTTTTSVoiceChanger:
    def __init__(self):
        # Audio settings
        self.CHUNK_SIZE = 1024 * 4  # Larger chunks for better STT accuracy
        self.SAMPLE_RATE = 16000  # Google STT requires 16kHz
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # ElevenLabs settings
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
        self.MODEL_ID = "eleven_multilingual_v2"  # Best for Russian
        
        # Google Cloud settings
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Audio queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # Control flags
        self.running = False
        self.audio = pyaudio.PyAudio()
        
        # Initialize Google clients
        self.speech_client = None
        self.tts_client = None
        self.init_google_clients()
        
        # WebSocket connection
        self.websocket = None
        
        print("🎤 Real-time STT→TTS Voice Changer initialized")
        print(f"🎵 Voice ID: {self.VOICE_ID}")
        print(f"🌍 Model: {self.MODEL_ID}")
        print(f"🔊 Sample Rate: {self.SAMPLE_RATE}Hz")
        print(f"📦 Chunk Size: {self.CHUNK_SIZE}")
    
    def init_google_clients(self):
        """Initialize Google Cloud Speech and TTS clients"""
        try:
            if self.GOOGLE_APPLICATION_CREDENTIALS:
                self.speech_client = speech.SpeechClient()
                self.tts_client = texttospeech.TextToSpeechClient()
                print("✅ Google Cloud clients initialized")
            else:
                print("⚠️  Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS")
                print("   You can get credentials from: https://console.cloud.google.com/apis/credentials")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud clients: {e}")
    
    async def connect_elevenlabs_websocket(self):
        """Connect to ElevenLabs WebSocket for streaming TTS"""
        try:
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            self.websocket = await websockets.connect(uri, extra_headers=headers)
            print("✅ Connected to ElevenLabs WebSocket")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to ElevenLabs WebSocket: {e}")
            return False
    
    async def send_text_to_elevenlabs(self, text):
        """Send text to ElevenLabs via WebSocket"""
        if not self.websocket:
            return
        
        try:
            # Send text chunk
            message = {
                "text": text,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            
            # Send end of stream marker
            await self.websocket.send(json.dumps({"text": ""}))
            
        except Exception as e:
            print(f"❌ Error sending text to ElevenLabs: {e}")
    
    async def receive_audio_from_elevenlabs(self):
        """Receive audio chunks from ElevenLabs WebSocket"""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if "audio" in data:
                    # Decode base64 audio
                    audio_data = base64.b64decode(data["audio"])
                    self.output_queue.put(audio_data)
                
                if data.get("isFinal"):
                    break
                    
        except Exception as e:
            print(f"❌ Error receiving audio from ElevenLabs: {e}")
    
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
                enable_word_time_offsets=False,
                model="latest_long"
            )
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract transcribed text
            transcribed_text = ""
            for result in response.results:
                if result.is_final:
                    transcribed_text += result.alternatives[0].transcript + " "
            
            return transcribed_text.strip() if transcribed_text else None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for input audio"""
        if self.running:
            self.input_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def output_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for output audio"""
        if self.running and not self.output_queue.empty():
            try:
                audio_data = self.output_queue.get_nowait()
                return (audio_data, pyaudio.paContinue)
            except queue.Empty:
                pass
        return (b'\x00' * frame_count * 2, pyaudio.paContinue)
    
    async def process_audio_loop(self):
        """Main audio processing loop"""
        print("🎵 Starting audio processing...")
        
        # Buffer for accumulating audio chunks
        audio_buffer = b""
        buffer_duration = 2.0  # Process every 2 seconds
        chunk_duration = self.CHUNK_SIZE / self.SAMPLE_RATE
        chunks_per_buffer = int(buffer_duration / chunk_duration)
        
        chunk_count = 0
        
        while self.running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.1)
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
                            
                            # Send to ElevenLabs
                            await self.send_text_to_elevenlabs(transcribed_text)
                            
                            # Receive and queue audio
                            await self.receive_audio_from_elevenlabs()
                        else:
                            print("🔇 No speech detected")
                    
                    # Reset buffer
                    audio_buffer = b""
                    chunk_count = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in audio processing: {e}")
                break
    
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
            
            # Output stream (speakers)
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,  # ElevenLabs output rate
                output=True,
                frames_per_buffer=1024,
                stream_callback=self.output_callback
            )
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            print("✅ Audio streams started")
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
            
            if hasattr(self, 'output_stream'):
                self.output_stream.stop_stream()
                self.output_stream.close()
            
            self.audio.terminate()
            print("✅ Audio streams stopped")
            
        except Exception as e:
            print(f"❌ Error stopping audio streams: {e}")
    
    async def run(self):
        """Main run method"""
        print("🚀 Starting real-time STT→TTS voice changer...")
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
            # Start audio processing
            await self.process_audio_loop()
            
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
    voice_changer = RealTimeSTTTTSVoiceChanger()
    
    try:
        asyncio.run(voice_changer.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 