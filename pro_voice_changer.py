#!/usr/bin/env python3
"""
Pro-Optimized Real-time Voice Changer using STT→TTS Pipeline
Leverages ElevenLabs Pro subscription benefits:
- 10 concurrent requests (vs 5 in Creator)
- 192 kbps audio quality (vs 128 kbps)
- 44.1kHz PCM audio output via API
- Turbo/Flash models for faster processing
- Priority processing
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
import concurrent.futures
from typing import List, Optional

# Load environment variables
load_dotenv()

class ProVoiceChanger:
    def __init__(self):
        # Audio settings optimized for Pro
        self.CHUNK_SIZE = 1024 * 4  # Larger chunks for better STT accuracy
        self.SAMPLE_RATE = 16000  # Google STT requires 16kHz
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # ElevenLabs Pro settings
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
        self.MODEL_ID = "eleven_turbo_v2"  # Pro: Use Turbo model for faster processing
        self.AUDIO_QUALITY = "192k"  # Pro: 192 kbps audio quality
        self.OUTPUT_FORMAT = "pcm_44100"  # Pro: 44.1kHz PCM output
        
        # Pro: Enhanced concurrency settings
        self.MAX_CONCURRENT_REQUESTS = 10  # Pro tier allows 10 concurrent requests
        self.REQUEST_SEMAPHORE = asyncio.Semaphore(10)
        
        # Google Cloud settings
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Audio queues with Pro optimizations
        self.input_queue = queue.Queue(maxsize=100)  # Larger buffer
        self.output_queue = queue.Queue(maxsize=200)  # Larger output buffer
        self.text_queue = queue.Queue(maxsize=50)
        
        # Pro: Multiple WebSocket connections for concurrency
        self.websocket_pool: List[websockets.WebSocketServerProtocol] = []
        self.websocket_lock = asyncio.Lock()
        
        # Control flags
        self.running = False
        self.audio = pyaudio.PyAudio()
        
        # Initialize Google clients
        self.speech_client = None
        self.tts_client = None
        self.init_google_clients()
        
        # Pro: Performance monitoring
        self.request_count = 0
        self.latency_stats = []
        
        print("🎤 Pro-Optimized Real-time STT→TTS Voice Changer initialized")
        print(f"🎵 Voice ID: {self.VOICE_ID}")
        print(f"⚡ Model: {self.MODEL_ID} (Turbo for faster processing)")
        print(f"🎧 Audio Quality: {self.AUDIO_QUALITY}")
        print(f"📊 Output Format: {self.OUTPUT_FORMAT}")
        print(f"🔄 Max Concurrent Requests: {self.MAX_CONCURRENT_REQUESTS}")
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
    
    async def get_websocket_connection(self):
        """Pro: Get a WebSocket connection from the pool or create new one"""
        async with self.websocket_lock:
            # Try to reuse existing connection
            for ws in self.websocket_pool:
                if not ws.closed:
                    return ws
            
            # Create new connection if needed
            if len(self.websocket_pool) < self.MAX_CONCURRENT_REQUESTS:
                try:
                    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                    params = {
                        "model_id": self.MODEL_ID,
                        "optimize_streaming_latency": "4",  # Pro: Optimized for speed
                        "output_format": self.OUTPUT_FORMAT,
                        "audio_quality": self.AUDIO_QUALITY
                    }
                    
                    # Build URI with parameters
                    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
                    uri = f"{uri}?{param_str}"
                    
                    headers = {
                        "xi-api-key": self.ELEVENLABS_API_KEY,
                        "Content-Type": "application/json"
                    }
                    
                    websocket = await websockets.connect(uri, extra_headers=headers)
                    self.websocket_pool.append(websocket)
                    print(f"✅ Created WebSocket connection #{len(self.websocket_pool)}")
                    return websocket
                    
                except Exception as e:
                    print(f"❌ Failed to create WebSocket connection: {e}")
                    return None
            
            # Wait for available connection
            return None
    
    async def send_text_to_elevenlabs_pro(self, text):
        """Pro: Send text to ElevenLabs with enhanced features"""
        start_time = time.time()
        
        async with self.REQUEST_SEMAPHORE:
            websocket = await self.get_websocket_connection()
            if not websocket:
                print("⚠️  No available WebSocket connections")
                return
            
            try:
                # Pro: Enhanced message with quality settings
                message = {
                    "text": text,
                    "try_trigger_generation": True,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                }
                
                await websocket.send(json.dumps(message))
                await websocket.send(json.dumps({"text": ""}))
                
                # Track performance
                self.request_count += 1
                latency = time.time() - start_time
                self.latency_stats.append(latency)
                
                print(f"⚡ Pro request #{self.request_count} sent (latency: {latency:.3f}s)")
                
            except Exception as e:
                print(f"❌ Error sending text to ElevenLabs: {e}")
                # Remove failed connection from pool
                if websocket in self.websocket_pool:
                    self.websocket_pool.remove(websocket)
    
    async def receive_audio_from_elevenlabs_pro(self):
        """Pro: Receive high-quality audio from ElevenLabs"""
        websocket = await self.get_websocket_connection()
        if not websocket:
            return
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if "audio" in data:
                    # Pro: Handle 44.1kHz PCM audio
                    audio_data = base64.b64decode(data["audio"])
                    
                    # Convert PCM to proper format if needed
                    if self.OUTPUT_FORMAT == "pcm_44100":
                        # PCM data is already in correct format for PyAudio
                        self.output_queue.put(audio_data)
                    else:
                        # Handle other formats
                        self.output_queue.put(audio_data)
                
                if data.get("isFinal"):
                    break
                    
        except Exception as e:
            print(f"❌ Error receiving audio from ElevenLabs: {e}")
            if websocket in self.websocket_pool:
                self.websocket_pool.remove(websocket)
    
    def transcribe_audio_pro(self, audio_chunk):
        """Pro: Enhanced transcription with better models"""
        if not self.speech_client:
            return None
        
        try:
            # Create audio content
            audio = speech.RecognitionAudio(content=audio_chunk)
            
            # Pro: Enhanced recognition config
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code="ru-RU",  # Russian
                enable_automatic_punctuation=True,
                enable_word_time_offsets=False,
                model="latest_long",  # Pro: Use latest model
                use_enhanced=True,  # Pro: Enhanced model for better accuracy
                enable_separate_recognition_per_channel=False,
                max_alternatives=1
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
            try:
                self.input_queue.put_nowait(in_data)
            except queue.Full:
                # Drop oldest frame if queue is full
                try:
                    self.input_queue.get_nowait()
                    self.input_queue.put_nowait(in_data)
                except queue.Empty:
                    pass
        return (None, pyaudio.paContinue)
    
    def output_callback(self, in_data, frame_count, time_info, status):
        """Pro: Enhanced output callback with better audio handling"""
        if self.running and not self.output_queue.empty():
            try:
                audio_data = self.output_queue.get_nowait()
                
                # Pro: Handle 44.1kHz PCM audio properly
                if len(audio_data) >= frame_count * 2:  # 16-bit = 2 bytes per sample
                    return (audio_data[:frame_count * 2], pyaudio.paContinue)
                else:
                    # Pad with silence if needed
                    padding = b'\x00' * (frame_count * 2 - len(audio_data))
                    return (audio_data + padding, pyaudio.paContinue)
                    
            except queue.Empty:
                pass
        return (b'\x00' * frame_count * 2, pyaudio.paContinue)
    
    async def process_audio_loop_pro(self):
        """Pro: Enhanced audio processing with better buffering"""
        print("🎵 Starting Pro audio processing...")
        
        # Pro: Optimized buffer settings
        audio_buffer = b""
        buffer_duration = 1.5  # Shorter buffer for faster response
        chunk_duration = self.CHUNK_SIZE / self.SAMPLE_RATE
        chunks_per_buffer = int(buffer_duration / chunk_duration)
        
        chunk_count = 0
        last_processing_time = time.time()
        
        while self.running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.05)  # Shorter timeout
                audio_buffer += audio_chunk
                chunk_count += 1
                
                current_time = time.time()
                
                # Process buffer when we have enough chunks or time has passed
                if (chunk_count >= chunks_per_buffer or 
                    current_time - last_processing_time >= buffer_duration):
                    
                    if audio_buffer:
                        print(f"🎤 Processing {len(audio_buffer)} bytes of audio...")
                        
                        # Transcribe audio
                        transcribed_text = self.transcribe_audio_pro(audio_buffer)
                        
                        if transcribed_text:
                            print(f"📝 Transcribed: '{transcribed_text}'")
                            
                            # Pro: Concurrent processing
                            asyncio.create_task(self.send_text_to_elevenlabs_pro(transcribed_text))
                            asyncio.create_task(self.receive_audio_from_elevenlabs_pro())
                        else:
                            print("🔇 No speech detected")
                    
                    # Reset buffer
                    audio_buffer = b""
                    chunk_count = 0
                    last_processing_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in audio processing: {e}")
                break
    
    def start_audio_streams_pro(self):
        """Pro: Enhanced audio streams with better quality"""
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
            
            # Pro: Output stream optimized for 44.1kHz PCM
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,  # Pro: 44.1kHz output
                output=True,
                frames_per_buffer=1024,
                stream_callback=self.output_callback
            )
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            print("✅ Pro audio streams started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start audio streams: {e}")
            return False
    
    def stop_audio_streams(self):
        """Stop PyAudio streams and clean up WebSocket pool"""
        try:
            if hasattr(self, 'input_stream'):
                self.input_stream.stop_stream()
                self.input_stream.close()
            
            if hasattr(self, 'output_stream'):
                self.output_stream.stop_stream()
                self.output_stream.close()
            
            self.audio.terminate()
            
            # Close all WebSocket connections
            for ws in self.websocket_pool:
                if not ws.closed:
                    asyncio.create_task(ws.close())
            
            print("✅ Audio streams and WebSocket connections stopped")
            
        except Exception as e:
            print(f"❌ Error stopping streams: {e}")
    
    def print_performance_stats(self):
        """Pro: Print performance statistics"""
        if self.latency_stats:
            avg_latency = sum(self.latency_stats) / len(self.latency_stats)
            min_latency = min(self.latency_stats)
            max_latency = max(self.latency_stats)
            
            print(f"\n📊 Pro Performance Statistics:")
            print(f"   Total Requests: {self.request_count}")
            print(f"   Average Latency: {avg_latency:.3f}s")
            print(f"   Min Latency: {min_latency:.3f}s")
            print(f"   Max Latency: {max_latency:.3f}s")
            print(f"   Active WebSocket Connections: {len(self.websocket_pool)}")
    
    async def run(self):
        """Main run method with Pro optimizations"""
        print("🚀 Starting Pro-Optimized Real-time STT→TTS voice changer...")
        print("📖 Speak in Russian and hear it transformed in real-time!")
        print("⚡ Pro features enabled: Turbo model, 192kbps, 44.1kHz PCM, 10 concurrent requests")
        print("⏹️  Press Ctrl+C to stop")
        
        # Start audio streams
        if not self.start_audio_streams_pro():
            return
        
        self.running = True
        
        try:
            # Start audio processing
            await self.process_audio_loop_pro()
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Pro voice changer...")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.running = False
            self.stop_audio_streams()
            self.print_performance_stats()
            print("👋 Pro voice changer stopped")

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
    
    # Create and run Pro voice changer
    voice_changer = ProVoiceChanger()
    
    try:
        asyncio.run(voice_changer.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 