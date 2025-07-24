#!/usr/bin/env python3
"""
Combined Speechmatics STT + ElevenLabs TTS - Word-Based Chunking with Parallel Connections
Simple parallel connection management to prevent connection overload
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
from typing import AsyncGenerator, Optional

# Import Speechmatics models
from speechmatics.rt import (
    AudioFormat,
    TranscriptionConfig,
)

# Load environment variables
load_dotenv()

# Configuration
SPEECHMATICS_API_KEY = os.getenv('SPEECHMATICS_API_KEY')
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Word chunking configuration
WORDS_PER_CHUNK = int(os.getenv("WORDS_PER_CHUNK", "3"))
MIN_WORDS_FOR_TIMEOUT = int(os.getenv("MIN_WORDS_FOR_TIMEOUT", "6"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "2.0"))
ENABLE_PUNCTUATION_BREAKS = os.getenv("ENABLE_PUNCTUATION_BREAKS", "true").lower() == "true"
PUNCTUATION_CHARS = list(os.getenv("PUNCTUATION_CHARS", ",.!?"))

# Connection management configuration - Safety measures to avoid bans
CONNECTION_REFRESH_INTERVAL = 31  # Create new connection every 31 seconds
CONNECTION_ESTABLISHMENT_TIMEOUT = 5  # Timeout for connection establishment

# Connection safety limits to prevent abuse detection
MIN_CONNECTION_LIFETIME = 30  # Minimum 30 seconds per connection
MAX_CONNECTIONS_PER_MINUTE = 3  # Max 3 connections per minute

# Debug: Print configuration values
print(f"🔧 Configuration loaded:")
print(f"   WORDS_PER_CHUNK: {WORDS_PER_CHUNK}")
print(f"   MIN_WORDS_FOR_TIMEOUT: {MIN_WORDS_FOR_TIMEOUT}")
print(f"   TIMEOUT_SECONDS: {TIMEOUT_SECONDS}")
print(f"   ENABLE_PUNCTUATION_BREAKS: {ENABLE_PUNCTUATION_BREAKS}")
print(f"   PUNCTUATION_CHARS: {PUNCTUATION_CHARS}")
print(f"   CONNECTION_REFRESH_INTERVAL: {CONNECTION_REFRESH_INTERVAL}s")
print(f"   MIN_CONNECTION_LIFETIME: {MIN_CONNECTION_LIFETIME}s")
print(f"   MAX_CONNECTIONS_PER_MINUTE: {MAX_CONNECTIONS_PER_MINUTE}")
print(f"   .env file path: {os.path.abspath('.env')}")
print(f"   .env file exists: {os.path.exists('.env')}")
print(f"   SPEECHMATICS_API_KEY exists: {bool(SPEECHMATICS_API_KEY)}")
print(f"   ELEVENLABS_API_KEY exists: {bool(EL_API_KEY)}")

if not SPEECHMATICS_API_KEY:
    print("ERROR: SPEECHMATICS_API_KEY not found in .env file")
    sys.exit(1)

if not EL_API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found in .env file")
    sys.exit(1)

# WebSocket URLs
SPEECHMATICS_URL = "wss://eu2.rt.speechmatics.com/v2"

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Speechmatics audio chunking
CHUNK_DURATION_MS = 50
BYTES_PER_SAMPLE = 2
SAMPLES_PER_CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

# Global state for transcript management
final_transcript = ""
current_partial = ""
audio_chunks_sent = 0
last_display_text = ""
last_tts_text = ""
last_final_transcript = ""
pending_tts_text = ""
last_tts_time = 0
tts_sent_length = 0

# Word-based chunking buffer
text_buffer = ""
last_buffer_update = 0
last_speech_activity = 0  # Track when we last received speech input
last_sent_chunk = ""  # Track the last chunk sent to TTS

# Echo loop detection
recent_transcripts = []  # Keep last 10 transcripts to detect loops
MAX_RECENT_TRANSCRIPTS = 10
echo_lockout_until = 0  # Prevent processing for a short time after echo detection

# Sequential TTS queue
tts_queue = []
tts_processing = False

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

# Global debug timer
debug_timer = DebugTimer()

class SpeechmaticsConnection:
    """Manages a single Speechmatics WebSocket connection"""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.established_time = None
        self.audio_chunks_sent = 0
        self.session_id = None
        self.receiver_task = None
        self.running = False
        
    async def connect(self):
        """Establish connection to Speechmatics"""
        try:
            debug_timer.mark(f"stt_connecting_{self.connection_id}", f"Connecting to Speechmatics (ID: {self.connection_id})")
            
            self.websocket = await websockets.connect(
                SPEECHMATICS_URL,
                additional_headers={"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"},
                ping_interval=30,
                ping_timeout=60
            )
            
            self.connected = True
            self.established_time = time.time()
            debug_timer.mark(f"stt_connected_{self.connection_id}", f"STT connected to Speechmatics (ID: {self.connection_id})")
            
            # Start recognition
            await self._start_recognition()
            
            # Start message receiver
            self.running = True
            self.receiver_task = asyncio.create_task(self._message_receiver())
            
        except Exception as e:
            print(f"❌ STT: Failed to connect {self.connection_id}: {e}")
            self.connected = False
            raise
    
    async def _start_recognition(self):
        """Start recognition on this connection"""
        # Configure audio settings
        audio_settings = AudioFormat(
            encoding="pcm_s16le",
            sample_rate=RATE
        )
        
        # Configure transcription settings
        transcription_config = TranscriptionConfig(
            language="ru",
            max_delay=0.7,
            max_delay_mode="flexible",
            operating_point="enhanced",
            enable_partials=True
        )
        
        # Start recognition
        start_message = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": audio_settings.encoding,
                "sample_rate": audio_settings.sample_rate
            },
            "transcription_config": transcription_config.to_dict()
        }
        
        await self.websocket.send(json.dumps(start_message))
        print(f"🎤 STT: Recognition started on {self.connection_id}")
    
    async def _message_receiver(self):
        """Background task to receive and process WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("message")
                    
                    if message_type == "RecognitionStarted":
                        self.session_id = data.get("id", "unknown")
                        debug_timer.mark(f"stt_recognition_started_{self.connection_id}", f"STT recognition started (ID: {self.session_id})")
                        
                    elif message_type == "AddTranscript":
                        transcript_segment = data.get("metadata", {}).get("transcript", "")
                        # Forward to global transcript handler
                        await self._handle_final_transcript(transcript_segment)
                        
                    elif message_type == "AddPartialTranscript":
                        full_partial = data.get("metadata", {}).get("transcript", "")
                        # Forward to global partial handler
                        await self._handle_partial_transcript(full_partial)
                        
                    elif message_type == "EndOfTranscript":
                        print(f"\n🎤 STT: End of transcript on {self.connection_id}")
                        break
                        
                    elif message_type == "Error":
                        error_type = data.get("type", "unknown")
                        error_reason = data.get("reason", "No reason provided")
                        print(f"\n❌ STT: Error on {self.connection_id} - {error_type}: {error_reason}")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"\n❌ STT: Failed to parse message on {self.connection_id}: {e}")
                    break
                except Exception as e:
                    print(f"\n❌ STT: Message handling error on {self.connection_id}: {e}")
                    break
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"\n❌ STT: Connection closed on {self.connection_id}: {e}")
        except Exception as e:
            print(f"\n❌ STT: Unexpected error on {self.connection_id}: {e}")
        finally:
            self.running = False
    
    async def _handle_final_transcript(self, transcript_segment):
        """Handle final transcript from this connection"""
        global final_transcript, current_partial
        final_transcript += transcript_segment
        current_partial = ""
        debug_timer.mark(f"stt_final_transcript_{self.connection_id}", f"Final transcript: {transcript_segment[:50]}...")
        print_transcript()
        
        # Send to TTS if we have accumulated substantial text
        if len(final_transcript.strip().split()) >= WORDS_PER_CHUNK:
            # Get TTS instance from global scope
            tts_instance = getattr(self, '_tts_instance', None)
            await send_to_tts_if_new(tts_instance, final_transcript + current_partial)
    
    async def _handle_partial_transcript(self, full_partial):
        """Handle partial transcript from this connection"""
        global final_transcript, current_partial
        
        if full_partial.startswith(final_transcript):
            new_partial = full_partial[len(final_transcript):]
            if new_partial != current_partial:
                current_partial = new_partial
                debug_timer.mark(f"stt_partial_transcript_{self.connection_id}", f"Partial transcript: {new_partial[:50]}...")
                print_transcript()
        else:
            if full_partial != current_partial:
                current_partial = full_partial
                debug_timer.mark(f"stt_partial_transcript_{self.connection_id}", f"Partial transcript: {full_partial[:50]}...")
                print_transcript()
    
    async def send_audio(self, audio_chunk):
        """Send audio chunk to this connection"""
        if self.connected and self.websocket:
            try:
                await self.websocket.send(audio_chunk)
                self.audio_chunks_sent += 1
            except Exception as e:
                print(f"❌ STT: Failed to send audio to {self.connection_id}: {e}")
                self.connected = False
    
    async def end_stream(self):
        """End the audio stream on this connection with proper acknowledgment"""
        if self.connected and self.websocket:
            try:
                end_message = {
                    "message": "EndOfStream",
                    "last_seq_no": self.audio_chunks_sent
                }
                await self.websocket.send(json.dumps(end_message))
                print(f"\n🎤 STT: End of stream sent on {self.connection_id} ({self.audio_chunks_sent} chunks)")
                
                # Wait for server acknowledgment (with timeout)
                await asyncio.wait_for(self._wait_for_end_ack(), timeout=5.0)
                
            except asyncio.TimeoutError:
                print(f"⚠️ Timeout waiting for EndOfStream acknowledgment on {self.connection_id}")
            except Exception as e:
                print(f"❌ STT: Failed to end stream on {self.connection_id}: {e}")
    
    async def _wait_for_end_ack(self):
        """Wait for server acknowledgment of EndOfStream"""
        # Wait a short time for any final messages
        await asyncio.sleep(0.5)
    
    async def close(self):
        """Close this connection gracefully with timeouts"""
        self.running = False
        
        # Cancel receiver task with timeout
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await asyncio.wait_for(self.receiver_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Close websocket gracefully with timeout
        if self.websocket:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=3.0)
            except asyncio.TimeoutError:
                print(f"⚠️ Force closing websocket on {self.connection_id}")
                self.websocket.close()
        
        self.connected = False
        print(f"🔌 STT: Connection {self.connection_id} closed")

class ConnectionManager:
    """Manages parallel Speechmatics connections"""
    
    def __init__(self, tts_instance=None):
        self.active_connection: Optional[SpeechmaticsConnection] = None
        self.backup_connection: Optional[SpeechmaticsConnection] = None
        self.connection_counter = 0
        self.last_refresh_time = 0
        self.running = False
        self.refresh_task = None
        self.tts_instance = tts_instance
        self.first_tts_time = None  # Track when first TTS response occurs
        self.refresh_cycle_started = False  # Track if refresh cycle has started
        
        # Connection rate limiting
        self.connection_timestamps = []  # Track connection creation times
        self.last_connection_time = 0  # Track last connection creation
    
    async def start(self):
        """Start the connection manager"""
        self.running = True
        
        # Create initial connection
        await self._create_active_connection()
        
        # Start background refresh task
        self.refresh_task = asyncio.create_task(self._connection_refresh_loop())
        
        debug_timer.mark("connection_manager_started", "Connection manager started")
    
    async def _create_active_connection(self):
        """Create a new active connection with rate limiting"""
        current_time = time.time()
        
        # Check rate limits
        if not self._can_create_connection(current_time):
            print(f"⚠️ Rate limit: Cannot create connection yet. Waiting...")
            await asyncio.sleep(5)  # Wait before retry
            return await self._create_active_connection()
        
        self.connection_counter += 1
        connection_id = f"conn_{self.connection_counter}"
        
        connection = SpeechmaticsConnection(connection_id)
        connection._tts_instance = self.tts_instance  # Pass TTS instance
        await connection.connect()
        
        self.active_connection = connection
        self.last_refresh_time = current_time
        self.last_connection_time = current_time
        self.connection_timestamps.append(current_time)
        
        debug_timer.mark(f"active_connection_created", f"Active connection {connection_id} created")
    
    async def _create_backup_connection(self):
        """Create a new backup connection in background with rate limiting"""
        current_time = time.time()
        
        # Check rate limits
        if not self._can_create_connection(current_time):
            print(f"⚠️ Rate limit: Cannot create backup connection yet. Skipping...")
            return
        
        self.connection_counter += 1
        connection_id = f"conn_{self.connection_counter}"
        
        try:
            connection = SpeechmaticsConnection(connection_id)
            connection._tts_instance = self.tts_instance  # Pass TTS instance
            await connection.connect()
            
            self.backup_connection = connection
            self.last_connection_time = current_time
            self.connection_timestamps.append(current_time)
            debug_timer.mark(f"backup_connection_created", f"Backup connection {connection_id} created")
            
        except Exception as e:
            print(f"❌ Failed to create backup connection: {e}")
            self.backup_connection = None
    
    async def _connection_refresh_loop(self):
        """Background loop to refresh connections"""
        while self.running:
            try:
                current_time = time.time()
                
                # Wait for first TTS response before starting refresh cycle
                if not self.refresh_cycle_started:
                    if self.first_tts_time is None:
                        # Still waiting for first TTS
                        await asyncio.sleep(1)
                        continue
                    else:
                        # First TTS occurred, start refresh cycle 10 seconds after it
                        time_since_first_tts = current_time - self.first_tts_time
                        if time_since_first_tts < 10:
                            # Still waiting for 10 seconds after first TTS
                            await asyncio.sleep(1)
                            continue
                        else:
                            # Start the refresh cycle
                            self.refresh_cycle_started = True
                            self.last_refresh_time = current_time
                            debug_timer.mark("refresh_cycle_started", f"Refresh cycle started {time_since_first_tts:.1f}s after first TTS")
                
                # Check if it's time to create a new backup connection
                if (current_time - self.last_refresh_time) >= CONNECTION_REFRESH_INTERVAL:
                    debug_timer.mark("connection_refresh_triggered", f"Creating backup connection after {CONNECTION_REFRESH_INTERVAL}s")
                    
                    # Close old backup connection if it exists
                    if self.backup_connection:
                        await self.backup_connection.close()
                    
                    # Create new backup connection
                    await self._create_backup_connection()
                    
                    # If backup connection is ready, swap it with active
                    if self.backup_connection and self.backup_connection.connected:
                        await self._swap_connections()
                    
                    self.last_refresh_time = current_time
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"❌ Connection refresh error: {e}")
                await asyncio.sleep(1)
    
    async def _swap_connections(self):
        """Swap backup connection to active"""
        if not self.backup_connection or not self.backup_connection.connected:
            return
        
        # Store old connection for cleanup
        old_connection = self.active_connection
        
        # Swap connections immediately
        self.active_connection = self.backup_connection
        self.backup_connection = None
        
        debug_timer.mark("connections_swapped", f"Swapped to new active connection {self.active_connection.connection_id}")
        
        # Clean up old connection in background (don't wait for it)
        if old_connection:
            asyncio.create_task(self._cleanup_old_connection(old_connection))
    
    def _can_create_connection(self, current_time):
        """Check if we can create a new connection based on rate limits"""
        # Clean old timestamps (older than 1 minute)
        self.connection_timestamps = [ts for ts in self.connection_timestamps 
                                    if current_time - ts < 60]
        
        # Check minimum time between connections
        if current_time - self.last_connection_time < MIN_CONNECTION_LIFETIME:
            return False
        
        # Check maximum connections per minute
        if len(self.connection_timestamps) >= MAX_CONNECTIONS_PER_MINUTE:
            return False
        
        return True
    
    async def _cleanup_old_connection(self, old_connection):
        """Clean up old connection in background"""
        try:
            await old_connection.end_stream()
            await old_connection.close()
        except Exception as e:
            print(f"❌ Error cleaning up old connection: {e}")
    
    async def send_audio(self, audio_chunk):
        """Send audio to the active connection"""
        if self.active_connection and self.active_connection.connected:
            await self.active_connection.send_audio(audio_chunk)
    
    def notify_first_tts(self):
        """Notify that first TTS response has occurred"""
        if self.first_tts_time is None:
            self.first_tts_time = time.time()
            debug_timer.mark("first_tts_detected", "First TTS response detected")
    
    async def stop(self):
        """Stop the connection manager"""
        self.running = False
        
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        if self.active_connection:
            await self.active_connection.close()
        
        if self.backup_connection:
            await self.backup_connection.close()
        
        debug_timer.mark("connection_manager_stopped", "Connection manager stopped")

class TTSAudioPlayer:
    """Dedicated audio player for TTS output"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.thread = None
    
    def start(self):
        """Start the audio player thread"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        debug_timer.mark("tts_player_started", "TTS Audio player started")
    
    def _run(self):
        """Audio player thread function"""
        try:
            print(f"🔊 TTS Player: Initializing with format={FORMAT}, channels={CHANNELS}, rate={RATE}, chunk_size={CHUNK_SIZE}")
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print(f"🔊 TTS Player: Stream initialized successfully")
            
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk:
                        print(f"🔊 TTS Player: Playing chunk: {len(chunk)} bytes")
                        self.stream.write(chunk)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ TTS Audio player error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Failed to initialize TTS audio player: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
    
    def play(self, audio_data):
        """Add audio data to the queue"""
        self.audio_queue.put(audio_data)
    
    def stop(self):
        """Stop the audio player"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

class ElevenLabsTTS:
    """Simple ElevenLabs TTS client"""
    
    def __init__(self, audio_player, connection_manager=None):
        self.voice_id = EL_VOICE_ID
        self.api_key = EL_API_KEY
        self.websocket = None
        self.audio_player = audio_player
        self.connected = False
        self.message_queue = asyncio.Queue()
        self.receiver_task = None
        self.running = False
        self.connection_manager = connection_manager
        self.first_audio_received = False
    
    async def connect(self):
        """Connect to ElevenLabs"""
        try:
            url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
            params = {
                "model_id": "eleven_flash_v2_5",
                "output_format": "pcm_16000",
                "auto_mode": "true",
                "inactivity_timeout": "60",
                "sync_alignment": "false"
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{query_string}"
            
            self.websocket = await websockets.connect(
                full_url,
                additional_headers={"xi-api-key": self.api_key}
            )
            debug_timer.mark("tts_connected", "TTS connected to ElevenLabs")
            
            self.context_id = f"conv_{uuid.uuid4().hex[:8]}"
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 0.7,
                    "speed": 1.1
                },
                "context_id": self.context_id
            }
            
            await self.websocket.send(json.dumps(init_message))
            self.connected = True
            debug_timer.mark("tts_initialized", "TTS context initialized")
            
            self.running = True
            self.receiver_task = asyncio.create_task(self._message_receiver())
            
        except Exception as e:
            print(f"❌ TTS: Failed to connect: {e}")
            raise
    
    async def _message_receiver(self):
        """Background task to receive and route WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.message_queue.put(data)
                except Exception as e:
                    print(f"❌ TTS: Message receiver error: {e}")
        except Exception as e:
            print(f"❌ TTS: Message receiver error: {e}")
        finally:
            self.running = False
    
    async def speak(self, text):
        """Speak a text phrase"""
        if not self.connected:
            return
        
        debug_timer.mark("tts_speak_start", f"TTS speaking: {text[:50]}...")
        
        text_message = {"text": text, "xi_api_key": self.api_key}
        await self.websocket.send(json.dumps(text_message))
        debug_timer.mark("tts_text_sent", "TTS text sent to ElevenLabs")
        
        try:
            audio_chunks_received = 0
            max_chunks = 15
            timeout = 8
            first_audio_received = False
            last_audio_time = time.time()
            start_time = time.time()
            
            while audio_chunks_received < max_chunks and (time.time() - start_time) < timeout:
                try:
                    data = await asyncio.wait_for(self.message_queue.get(), timeout=0.05)
                    
                    if "audio" in data:
                        audio_bytes = base64.b64decode(data["audio"])
                        
                        if not first_audio_received:
                            debug_timer.mark("tts_first_audio", "TTS first audio chunk received")
                            first_audio_received = True
                            
                            # Notify connection manager of first TTS response
                            if not self.first_audio_received and self.connection_manager:
                                self.first_audio_received = True
                                self.connection_manager.notify_first_tts()
                        
                        self.audio_player.play(audio_bytes)
                        audio_chunks_received += 1
                        last_audio_time = time.time()
                        
                        if data.get("is_final", False):
                            print(f"🔊 TTS: Audio complete ({audio_chunks_received} chunks)")
                            break
                    
                except asyncio.TimeoutError:
                    if (time.time() - last_audio_time) > 2.0:
                        print(f"🔊 TTS: Audio gap timeout after 2s")
                        break
                    
                    if (time.time() - start_time) >= timeout:
                        print(f"🔊 TTS: Audio timeout after {timeout}s")
                        break
                    continue
                except Exception as e:
                    print(f"❌ TTS: Error processing audio: {e}")
                    break
            
            if audio_chunks_received == 0:
                print(f"🔊 TTS: No audio received")
                    
        except Exception as e:
            print(f"❌ TTS: Error receiving audio: {e}")
    
    async def close(self):
        """Close the connection"""
        self.running = False
        
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()

def print_transcript():
    """Print current transcript with real-time updates"""
    global final_transcript, current_partial, last_display_text
    display_text = final_transcript + current_partial
    
    if display_text != last_display_text:
        print(f"\r🎤 STT: {display_text}", end="", flush=True)
        last_display_text = display_text

async def send_to_tts_if_new(tts, full_text):
    """Word-based chunking with timeout fallback"""
    global text_buffer, last_buffer_update, last_tts_text, tts_sent_length, last_speech_activity, recent_transcripts, echo_lockout_until
    
    # Check if we're in echo lockout period
    if time.time() < echo_lockout_until:
        debug_timer.mark("echo_lockout", f"Echo lockout active, ignoring input")
        return
    
    new_text = full_text[tts_sent_length:].strip()
    if new_text:
        # Check for echo loop - if this text contains recent TTS output
        if is_echo_loop(new_text):
            debug_timer.mark("echo_detected", f"Echo loop detected, ignoring: {new_text[:50]}...")
            # Clear buffer and set lockout period
            text_buffer = ""
            last_buffer_update = 0
            echo_lockout_until = time.time() + 3.0  # Lockout for 3 seconds
            return
        
        if text_buffer and not text_buffer.endswith(' '):
            text_buffer += ' '
        text_buffer += new_text + " "
        last_buffer_update = time.time()
        last_speech_activity = time.time()  # Update speech activity
        debug_timer.mark("buffer_updated", f"Buffer updated: {len(text_buffer)} chars")
    
    await process_text_buffer(tts)
    
    tts_sent_length = len(full_text)
    last_tts_text = full_text

async def process_text_buffer(tts):
    """Process text buffer using hybrid word-based + punctuation chunking"""
    global text_buffer, last_buffer_update, last_tts_time, tts_queue, tts_processing, last_speech_activity
    
    if not text_buffer.strip():
        return
    
    words = text_buffer.strip().split()
    chunks = []
    current_chunk_words = []
    
    for i, word in enumerate(words):
        current_chunk_words.append(word)
        
        should_break = False
        
        if ENABLE_PUNCTUATION_BREAKS and len(current_chunk_words) >= 2:
            for punct in PUNCTUATION_CHARS:
                if word.endswith(punct):
                    should_break = True
                    break
        
        if not should_break and len(current_chunk_words) >= WORDS_PER_CHUNK:
            should_break = True
        
        if should_break:
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            current_chunk_words = []
    
    for chunk in chunks:
        if chunk and len(chunk.split()) >= 2:
            # Check if this chunk is different from what we last sent and not an echo
            if chunk != last_sent_chunk and not is_echo_loop(chunk):
                tts_queue.append(chunk)
                debug_timer.mark("tts_queued", f"Queued hybrid chunk ({len(chunk.split())} words): {chunk[:50]}...")
            elif is_echo_loop(chunk):
                debug_timer.mark("chunk_echo_detected", f"Chunk echo detected, skipping: {chunk[:50]}...")
            else:
                debug_timer.mark("chunk_skipped", f"Skipped duplicate chunk: {chunk[:50]}...")
    
    text_buffer = " ".join(current_chunk_words) + " " if current_chunk_words else ""
    
    # Timeout logic - only process if we have enough words and haven't already sent this text
    if text_buffer.strip() and (time.time() - last_buffer_update) > TIMEOUT_SECONDS:
        remaining_text = text_buffer.strip()
        if len(remaining_text.split()) >= MIN_WORDS_FOR_TIMEOUT:
            # Check if this text is different from what we last sent and not an echo
            if remaining_text != last_sent_chunk and not is_echo_loop(remaining_text):
                tts_queue.append(remaining_text)
                debug_timer.mark("tts_queued", f"Queued timeout chunk ({len(remaining_text.split())} words): {remaining_text[:50]}...")
                text_buffer = ""
            elif is_echo_loop(remaining_text):
                # Clear buffer if echo detected to prevent loops
                debug_timer.mark("buffer_cleared", f"Cleared echo text to prevent loop: {remaining_text[:50]}...")
                text_buffer = ""
            else:
                # Clear buffer if it's the same text to prevent loops
                debug_timer.mark("buffer_cleared", f"Cleared duplicate text to prevent loop: {remaining_text[:50]}...")
                text_buffer = ""
    
    # Clear buffer if no speech activity for extended period (5 seconds)
    if text_buffer.strip() and (time.time() - last_speech_activity) > 5.0:
        debug_timer.mark("buffer_cleared", f"Cleared buffer due to no speech activity: {text_buffer.strip()[:50]}...")
        text_buffer = ""
    
    if tts_queue and not tts_processing:
        await process_tts_queue(tts)

async def process_tts_queue(tts):
    """Process TTS queue sequentially"""
    global tts_queue, tts_processing, last_tts_time, last_sent_chunk, recent_transcripts
    
    if tts_processing:
        return
    
    tts_processing = True
    
    try:
        while tts_queue:
            chunk = tts_queue.pop(0)
            debug_timer.mark("tts_sending", f"Sending queued chunk ({len(chunk.split())} words): {chunk[:50]}...")
            await tts.speak(chunk)
            last_tts_time = time.time()
            last_sent_chunk = chunk  # Track the last chunk sent
            
            # Add to recent transcripts for echo detection
            recent_transcripts.append(chunk)
            if len(recent_transcripts) > MAX_RECENT_TRANSCRIPTS:
                recent_transcripts.pop(0)
            
            debug_timer.mark("tts_sent", f"Queued chunk sent to TTS ({len(chunk)} chars)")
            
            await asyncio.sleep(0.1)
    
    finally:
        tts_processing = False

def is_echo_loop(new_text):
    """Check if new text is an echo of recent TTS output"""
    if not recent_transcripts:
        return False
    
    # Check if new text contains any recent TTS chunks
    new_text_lower = new_text.lower()
    for chunk in recent_transcripts[-3:]:  # Check last 3 chunks
        chunk_lower = chunk.lower()
        if len(chunk_lower) > 10 and chunk_lower in new_text_lower:
            return True
    
    # Check for repeated phrases
    words = new_text.split()
    if len(words) > 3:
        # Look for repeated word sequences
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if new_text.count(phrase) > 1:
                return True
    
    return False

async def mic_stream_generator() -> AsyncGenerator[bytes, None]:
    """Generate audio chunks from microphone"""
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=SAMPLES_PER_CHUNK
        )
        
        print("🎤 STT: Microphone started. Speak now...")
        
        while True:
            try:
                audio_data = stream.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
                
                if len(audio_data) % BYTES_PER_SAMPLE != 0:
                    padding_needed = BYTES_PER_SAMPLE - (len(audio_data) % BYTES_PER_SAMPLE)
                    audio_data += b'\x00' * padding_needed
                
                yield audio_data
                await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)
                
            except Exception as e:
                print(f"❌ STT: Microphone error: {e}")
                break
                
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

async def audio_sender_task(connection_manager):
    """Task to send audio to the active connection"""
    audio_generator = mic_stream_generator()
    async for audio_chunk in audio_generator:
        await connection_manager.send_audio(audio_chunk)

async def tts_monitor_task(tts):
    """Monitor TTS system"""
    try:
        debug_timer.mark("tts_monitor_started", "TTS monitor started")
        
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"❌ TTS: Monitor failed: {e}")

async def flush_pending_tts(tts):
    """Flush any remaining text in buffer and queue"""
    global text_buffer, tts_queue
    
    if text_buffer.strip():
        remaining_text = text_buffer.strip()
        if len(remaining_text.split()) >= 2:
            tts_queue.append(remaining_text)
            debug_timer.mark("tts_flush", f"Flushing buffer to queue: {remaining_text[:50]}...")
            text_buffer = ""
    
    if tts_queue:
        await process_tts_queue(tts)
        debug_timer.mark("tts_flushed", "Queue flushed")

async def main():
    """Main function"""
    print("🚀 Starting Combined STT + TTS System - Parallel Connection Management")
    print(f"📋 STT: Real-time transcription with parallel connections")
    print(f"🔊 TTS: Parallel audio output")
    print(f"📝 Chunking: {WORDS_PER_CHUNK} words per chunk")
    if ENABLE_PUNCTUATION_BREAKS:
        print(f"🔤 Punctuation breaks: {PUNCTUATION_CHARS}")
    else:
        print(f"🔤 Punctuation breaks: disabled")
    print(f"⏱️  Timeout: {TIMEOUT_SECONDS}s (min {MIN_WORDS_FOR_TIMEOUT} words)")
    print(f"🔄 Connection refresh: every {CONNECTION_REFRESH_INTERVAL}s")
    
    # Create TTS audio player
    tts_player = TTSAudioPlayer()
    tts_player.start()
    
    # Create connection manager first
    connection_manager = ConnectionManager()
    
    # Create TTS client with connection manager reference
    tts = ElevenLabsTTS(tts_player, connection_manager)
    
    # Update connection manager with TTS instance
    connection_manager.tts_instance = tts
    
    try:
        # Connect TTS
        await tts.connect()
        
        # Start connection manager
        await connection_manager.start()
        
        # Start TTS monitor task
        tts_task = asyncio.create_task(tts_monitor_task(tts))
        
        # Start audio sender task
        audio_task = asyncio.create_task(audio_sender_task(connection_manager))
        
        # Wait for all tasks
        done, pending = await asyncio.wait(
            [audio_task, tts_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cleanup
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        await flush_pending_tts(tts)
        await connection_manager.stop()
        await tts.close()
        tts_player.stop()
        debug_timer.mark("system_shutdown", "System shutdown complete")
        
        # Print debug summary
        debug_timer.summary()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 