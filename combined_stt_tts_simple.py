#!/usr/bin/env python3
"""
Combined Speechmatics STT + ElevenLabs TTS - Simple Parallel Version
Runs STT transcription and TTS playback simultaneously
"""

import asyncio
import base64
import json
import os
import sys
import time
import threading
import queue
import uuid
import websockets
import pyaudio
from dotenv import load_dotenv
from typing import AsyncGenerator
import base64

# Import Speechmatics models
from speechmatics.models import (
    AudioSettings,
    TranscriptionConfig,
)

# Load environment variables
load_dotenv()

# Configuration
SPEECHMATICS_API_KEY = os.getenv('SPEECHMATICS_API_KEY')
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

if not SPEECHMATICS_API_KEY:
    print("ERROR: SPEECHMATICS_API_KEY not found in .env file")
    sys.exit(1)

if not EL_API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found in .env file")
    sys.exit(1)

# WebSocket URLs
SPEECHMATICS_URL = "wss://eu2.rt.speechmatics.com/v2"
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/text-to-speech"

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Speechmatics audio chunking
CHUNK_DURATION_MS = 100
BYTES_PER_SAMPLE = 2
SAMPLES_PER_CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

# Global state for transcript management
final_transcript = ""
current_partial = ""
audio_chunks_sent = 0
last_display_text = ""

class DebugTimer:
    """Debug timer for tracking operation timings"""
    
    def __init__(self):
        self.start_time = time.time()
        self.events = []
    
    def mark(self, event_name, description=""):
        """Mark an event with timestamp"""
        timestamp = time.time() - self.start_time
        self.events.append({
            'time': timestamp,
            'event': event_name,
            'description': description
        })
        print(f"⏱️  [{timestamp:.3f}s] {event_name}: {description}")
    
    def summary(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("📈 DEBUG TIMING SUMMARY")
        print("="*60)
        
        for i, event in enumerate(self.events):
            if i > 0:
                prev_time = self.events[i-1]['time']
                interval = (event['time'] - prev_time) * 1000
                print(f"⏱️  [{event['time']:.3f}s] {event['event']} (+{interval:.1f}ms)")
            else:
                print(f"⏱️  [{event['time']:.3f}s] {event['event']}")
        
        print("="*60)

class OptimizedElevenLabsTTS:
    """Optimized ElevenLabs TTS with real-time streaming"""
    
    def __init__(self, voice_id=None, api_key=None):
        self.voice_id = voice_id or EL_VOICE_ID
        self.api_key = api_key or EL_API_KEY
        self.websocket = None
        self.connected = False
        self.context_id = None
        self.audio_queue = queue.Queue()
        self.player = None
        self.player_thread = None
        self.message_queue = asyncio.Queue()
        self.receiver_task = None
        self.running = False
        self.timer = DebugTimer()
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
    
    def start_audio_player(self):
        """Start the audio player thread"""
        self.player = SimpleAudioPlayer(self.audio_queue)
        self.player_thread = threading.Thread(target=self.player.run, daemon=True)
        self.player_thread.start()
        self.timer.mark("audio_player_started", "Audio player thread started")
    
    async def _message_receiver(self):
        """Background task to receive and route WebSocket messages"""
        try:
            async for message in self.websocket:
                await self.message_queue.put(message)
        except Exception as e:
            print(f"❌ Message receiver error: {e}")
        finally:
            self.running = False
    
    async def connect(self):
        """Establish WebSocket connection to ElevenLabs"""
        self.timer.mark("connection_start", "Starting WebSocket connection")
        
        try:
            self.websocket = await websockets.connect(
                f"{ELEVENLABS_URL}/{self.voice_id}/multi-stream-input",
                extra_headers={"xi-api-key": self.api_key}
            )
            self.connected = True
            self.timer.mark("websocket_connected", "WebSocket connection established")
            
            # Initialize context
            self.context_id = str(uuid.uuid4())
            init_message = {
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                "xi_api_key": self.api_key,
                "generation_config": {"chunk_length_schedule": [50, 90, 120, 150, 200]},
                "model_id": "eleven_multilingual_v2"
            }
            
            await self.websocket.send(json.dumps(init_message))
            self.timer.mark("context_initialized", f"Context initialized: {self.context_id}")
            
            # Start message receiver
            self.running = True
            self.receiver_task = asyncio.create_task(self._message_receiver())
            
            self.timer.mark("ready_for_speech", "Ready for real-time speech")
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    async def speak_realtime(self, text, phrase_id=""):
        """Send text for real-time TTS"""
        if not self.connected:
            raise RuntimeError("Not connected to ElevenLabs")
        
        self.timer.mark(f"speech_start_{phrase_id}", f"Starting speech: '{text[:30]}...'")
        
        # Send text message
        text_message = {"text": text, "xi_api_key": self.api_key}
        await self.websocket.send(json.dumps(text_message))
        self.timer.mark(f"text_sent_{phrase_id}", "Text sent to ElevenLabs")
        
        # Start audio receiver
        audio_task = asyncio.create_task(self._receive_audio_background(phrase_id))
        
        # Wait for audio or timeout
        try:
            await asyncio.wait_for(audio_task, timeout=5.0)
        except asyncio.TimeoutError:
            self.timer.mark(f"audio_gap_timeout_{phrase_id}", "Audio gap timeout after 5s")
    
    async def _receive_audio_background(self, phrase_id=""):
        """Background task to receive audio chunks"""
        first_audio_received = False
        
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=2.0)
                    data = json.loads(message)
                    
                    if data.get("audio"):
                        # Decode base64 audio
                        audio_data = base64.b64decode(data["audio"])
                        self.audio_queue.put(audio_data)
                        
                        if not first_audio_received:
                            self.timer.mark(f"first_audio_{phrase_id}", "First audio chunk received")
                            first_audio_received = True
                    
                    elif data.get("isFinal"):
                        # End of audio stream
                        break
                        
                except asyncio.TimeoutError:
                    if first_audio_received:
                        break
                    else:
                        self.timer.mark(f"no_first_audio_{phrase_id}", "No first audio received")
                        break
                        
        except Exception as e:
            print(f"❌ Audio receiver error: {e}")
    
    async def speak_multiple_realtime(self, texts):
        """Send multiple texts with optimized rate limiting"""
        self.timer.mark("multi_speech_start", f"Starting {len(texts)} phrases with optimized rate limiting")
        
        tasks = []
        for i, text in enumerate(texts, 1):
            task = asyncio.create_task(self.speak_realtime(text, f"phrase_{i}"))
            tasks.append(task)
            # Small delay between phrases to prevent overwhelming
            await asyncio.sleep(0.1)
        
        self.timer.mark("all_phrases_sent", "All phrases sent with optimized rate limiting")
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        self.timer.mark("multi_speech_complete", "All phrases completed")
    
    async def close(self):
        """Close the connection and cleanup"""
        self.timer.mark("closing_start", "Starting cleanup")
        
        self.running = False
        
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.timer.mark("connection_closed", "WebSocket connection closed")
        
        if self.player:
            self.player.stop()
            if self.player_thread:
                self.player_thread.join(timeout=1)
        
        self.timer.mark("cleanup_complete", "All cleanup completed")

class SimpleAudioPlayer:
    """Simple audio player for real-time playback"""
    
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
    
    def run(self):
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk:
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

def print_current_transcript():
    """Print the current transcript only if it has changed"""
    global final_transcript, current_partial, last_display_text
    display_text = final_transcript + current_partial
    
    if display_text != last_display_text:
        print(f"\r🎤 STT: {display_text}", end="", flush=True)
        last_display_text = display_text

async def mic_stream_generator() -> AsyncGenerator[bytes, None]:
    """Asynchronous generator that yields audio chunks from the microphone"""
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=SAMPLES_PER_CHUNK
        )
        
        print("🎤 INFO: Microphone stream started. Speak now...")
        
        while True:
            try:
                audio_data = stream.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
                
                if len(audio_data) % BYTES_PER_SAMPLE != 0:
                    padding_needed = BYTES_PER_SAMPLE - (len(audio_data) % BYTES_PER_SAMPLE)
                    audio_data += b'\x00' * padding_needed
                
                yield audio_data
                
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)
                
            except Exception as e:
                print(f"❌ Microphone error: {e}")
                break
                
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

async def speechmatics_receive_handler(websocket):
    """Handle incoming messages from Speechmatics server"""
    global final_transcript, current_partial
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("message")
                
                if message_type == "RecognitionStarted":
                    session_id = data.get("id", "unknown")
                    print(f"\n🎤 INFO: Recognition started with session ID: {session_id}")
                    
                elif message_type == "AddTranscript":
                    transcript_segment = data.get("metadata", {}).get("transcript", "")
                    final_transcript += transcript_segment
                    current_partial = ""
                    print_current_transcript()
                    
                elif message_type == "AddPartialTranscript":
                    full_partial = data.get("metadata", {}).get("transcript", "")
                    
                    if full_partial.startswith(final_transcript):
                        new_partial = full_partial[len(final_transcript):]
                        if new_partial != current_partial:
                            current_partial = new_partial
                            print_current_transcript()
                    else:
                        if full_partial != current_partial:
                            current_partial = full_partial
                            print_current_transcript()
                    
                elif message_type == "EndOfTranscript":
                    print("\n🎤 INFO: End of transcript received.")
                    break
                    
                elif message_type == "Error":
                    error_type = data.get("type", "unknown")
                    error_reason = data.get("reason", "No reason provided")
                    print(f"\n❌ ERROR: Server error - Type: {error_type}, Reason: {error_reason}")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ ERROR: Failed to parse server message: {e}")
                break
            except Exception as e:
                print(f"\n❌ ERROR: Unexpected error in message handling: {e}")
                break
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ ERROR: WebSocket connection closed: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error in receive handler: {e}")

async def speechmatics_send_handler(websocket):
    """Handle sending audio to Speechmatics server"""
    global audio_chunks_sent
    
    # Configure audio settings
    audio_settings = AudioSettings(
        encoding="pcm_s16le",
        sample_rate=RATE
    )
    
    # Configure transcription settings
    transcription_config = TranscriptionConfig(
        language="ru",
        max_delay=1,
        max_delay_mode="flexible",
        accuracy="enhanced"
    )
    
    # Create the StartRecognition message
    start_recognition_message = {
        "message": "StartRecognition",
        "audio_format": {
            "type": "raw",
            "encoding": audio_settings.encoding,
            "sample_rate": audio_settings.sample_rate
        },
        "transcription_config": transcription_config.asdict()
    }
    
    try:
        await websocket.send(json.dumps(start_recognition_message))
        print("🎤 INFO: StartRecognition message sent. Waiting for confirmation...")
        
        audio_generator = mic_stream_generator()
        async for audio_chunk in audio_generator:
            try:
                await websocket.send(audio_chunk)
                audio_chunks_sent += 1
            except websockets.exceptions.ConnectionClosed:
                print("❌ ERROR: Connection closed by server while sending audio.")
                break
            except Exception as e:
                print(f"❌ ERROR: Failed to send audio chunk: {e}")
                break
        
        end_of_stream_message = {
            "message": "EndOfStream",
            "last_seq_no": audio_chunks_sent
        }
        await websocket.send(json.dumps(end_of_stream_message))
        print(f"\n🎤 INFO: EndOfStream message sent with last_seq_no: {audio_chunks_sent}")
        
    except Exception as e:
        print(f"❌ ERROR: Unexpected error in send handler: {e}")

async def tts_test_task():
    """Task to run TTS tests in parallel with STT"""
    print("\n🔊 Starting TTS tests in parallel...")
    
    tts = OptimizedElevenLabsTTS()
    
    try:
        tts.start_audio_player()
        await tts.connect()
        
        # Wait a bit for STT to start
        await asyncio.sleep(2)
        
        # Test phrases
        test_phrases = [
            "Привет, это тест системы TTS в реальном времени.",
            "Вторая фраза: Система работает параллельно со STT.",
            "Третья фраза: Оба сервиса работают одновременно.",
            "Четвертая фраза: Тест завершен успешно."
        ]
        
        print("\n🔊 TTS: Starting test phrases...")
        await tts.speak_multiple_realtime(test_phrases)
        
        print("\n🔊 TTS: Tests completed!")
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
    finally:
        await tts.close()

async def main():
    """Main function running STT and TTS in parallel"""
    print("🚀 Starting combined STT + TTS system...")
    print("📋 STT will show transcription on screen")
    print("🔊 TTS will play test phrases in parallel")
    
    try:
        # Start TTS test task
        tts_task = asyncio.create_task(tts_test_task())
        
        # Start Speechmatics STT
        print("\n🎤 Connecting to Speechmatics Real-Time API...")
        
        async with websockets.connect(
            SPEECHMATICS_URL,
            extra_headers={"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"},
            ping_interval=30,
            ping_timeout=60
        ) as websocket:
            print("🎤 INFO: WebSocket connection established successfully.")
            
            # Run STT sender and receiver concurrently
            receiver_task = asyncio.create_task(speechmatics_receive_handler(websocket))
            sender_task = asyncio.create_task(speechmatics_send_handler(websocket))
            
            # Wait for all tasks to complete
            done, pending = await asyncio.wait(
                [receiver_task, sender_task, tts_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Clean up pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ ERROR: Connection failed. Status code: {e.status_code}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"❌ ERROR: Connection closed unexpectedly: {e}")
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERROR: Fatal error: {e}")
        sys.exit(1) 