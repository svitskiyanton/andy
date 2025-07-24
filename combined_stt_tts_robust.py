#!/usr/bin/env python3
"""
Combined Speechmatics STT + ElevenLabs TTS - Robust Single Connection
Single resilient connection with keepalives and automatic reconnection
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
import random
from dotenv import load_dotenv
from typing import AsyncGenerator

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
WORDS_PER_CHUNK = int(os.getenv("WORDS_PER_CHUNK", "3"))  # Default: 3 words per chunk
MIN_WORDS_FOR_TIMEOUT = int(os.getenv("MIN_WORDS_FOR_TIMEOUT", "6"))  # Minimum words for timeout
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "1.0"))  # Reduced from 2.0 to 1.0 for faster processing
ENABLE_PUNCTUATION_BREAKS = os.getenv("ENABLE_PUNCTUATION_BREAKS", "true").lower() == "true"  # Enable punctuation breaks
PUNCTUATION_CHARS = list(os.getenv("PUNCTUATION_CHARS", ",.!?"))  # Punctuation that breaks chunks

# Reconnection configuration
INITIAL_RECONNECT_DELAY = 1.0  # Initial delay in seconds
MAX_RECONNECT_DELAY = 60.0      # Maximum delay in seconds
RECONNECT_JITTER = True         # Add random jitter to prevent thundering herd

# Health monitoring configuration
HEALTH_CHECK_INTERVAL = 3.0     # Check system health every 3 seconds
MAX_SESSION_DURATION = 300.0    # Maximum session duration (5 minutes)
TTS_QUEUE_TIMEOUT = 5.0         # Maximum time for TTS queue processing (5 seconds)
CONNECTION_HEALTH_TIMEOUT = 3.0  # Maximum time without activity (3 seconds)

# Debug: Print configuration values will be moved after all variables are defined

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

# Speechmatics audio chunking - Optimized for faster response
CHUNK_DURATION_MS = 50  # Reduced from 100ms to 50ms for faster processing
BYTES_PER_SAMPLE = 2
SAMPLES_PER_CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

# Silence detection for faster end-of-speech
SILENCE_THRESHOLD = 0.005  # More sensitive audio level threshold
SILENCE_DURATION_MS = 3000  # Consider silence after 3 seconds (more natural)
SILENCE_CHUNKS = int(SILENCE_DURATION_MS / CHUNK_DURATION_MS)  # Number of silent chunks

# Global state for transcript management
final_transcript = ""
current_partial = ""
audio_chunks_sent = 0
last_display_text = ""
last_tts_text = ""  # Track last text sent to TTS
last_final_transcript = ""  # Track last final transcript sent to TTS
pending_tts_text = ""  # Accumulate text for TTS
last_tts_time = 0  # Track when last TTS was sent
tts_sent_length = 0  # Track how much text we've already sent to TTS

# Word-based chunking buffer
text_buffer = ""  # Buffer for accumulating text
last_buffer_update = 0  # Track when buffer was last updated

# Sequential TTS queue
tts_queue = []  # Queue of chunks to send in order
tts_processing = False  # Flag to prevent concurrent processing

# Health monitoring variables
last_activity_time = time.time()  # Track last system activity
session_start_time = time.time()  # Track session start time
health_check_task = None  # Health monitoring task
system_healthy = True  # System health flag

# Debug: Print configuration values
print(f"🔧 Configuration loaded:")
print(f"   WORDS_PER_CHUNK: {WORDS_PER_CHUNK}")
print(f"   MIN_WORDS_FOR_TIMEOUT: {MIN_WORDS_FOR_TIMEOUT}")
print(f"   TIMEOUT_SECONDS: {TIMEOUT_SECONDS}")
print(f"   ENABLE_PUNCTUATION_BREAKS: {ENABLE_PUNCTUATION_BREAKS}")
print(f"   PUNCTUATION_CHARS: {PUNCTUATION_CHARS}")
print(f"   SILENCE_THRESHOLD: {SILENCE_THRESHOLD}")
print(f"   SILENCE_DURATION_MS: {SILENCE_DURATION_MS}")
print(f"   INITIAL_RECONNECT_DELAY: {INITIAL_RECONNECT_DELAY}s")
print(f"   MAX_RECONNECT_DELAY: {MAX_RECONNECT_DELAY}s")
print(f"   HEALTH_CHECK_INTERVAL: {HEALTH_CHECK_INTERVAL}s")
print(f"   MAX_SESSION_DURATION: {MAX_SESSION_DURATION}s")
print(f"   CONNECTION_HEALTH_TIMEOUT: {CONNECTION_HEALTH_TIMEOUT}s")
print(f"   .env file path: {os.path.abspath('.env')}")
print(f"   .env file exists: {os.path.exists('.env')}")
print(f"   SPEECHMATICS_API_KEY exists: {bool(SPEECHMATICS_API_KEY)}")
print(f"   ELEVENLABS_API_KEY exists: {bool(EL_API_KEY)}")

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
    
    def __init__(self, audio_player):
        self.voice_id = EL_VOICE_ID
        self.api_key = EL_API_KEY
        self.websocket = None
        self.audio_player = audio_player
        self.connected = False
        self.message_queue = asyncio.Queue()
        self.receiver_task = None
        self.running = False
    
    async def connect(self):
        """Connect to ElevenLabs"""
        try:
            # Connect with proper PCM output format
            url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
            params = {
                "model_id": "eleven_flash_v2_5",
                "output_format": "pcm_16000",  # Force PCM output
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
            
            # Initialize context with optimized settings
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
            
            # Start message receiver
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
        
        # Send text
        text_message = {"text": text, "xi_api_key": self.api_key}
        await self.websocket.send(json.dumps(text_message))
        debug_timer.mark("tts_text_sent", "TTS text sent to ElevenLabs")
        
        # Receive audio using message queue
        try:
            audio_chunks_received = 0
            max_chunks = 15
            timeout = 4  # Reduced from 8 to 4 seconds
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
                        
                        # Send raw audio bytes to player (no processing)
                        self.audio_player.play(audio_bytes)
                        audio_chunks_received += 1
                        last_audio_time = time.time()
                        
                        if data.get("is_final", False):
                            print(f"🔊 TTS: Audio complete ({audio_chunks_received} chunks)")
                            break
                    
                except asyncio.TimeoutError:
                    if (time.time() - last_audio_time) > 1.0:  # Reduced from 2.0 to 1.0 seconds
                        print(f"🔊 TTS: Audio gap timeout after 1s")
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
        # Use real-time display like the fast version
        print(f"\r🎤 STT: {display_text}", end="", flush=True)
        last_display_text = display_text

async def send_to_tts_if_new(tts, full_text):
    """Word-based chunking with timeout fallback"""
    global text_buffer, last_buffer_update, last_tts_text, tts_sent_length
    
    # Update activity timestamp
    update_activity()
    
    # Update buffer with new text
    new_text = full_text[tts_sent_length:].strip()
    if new_text:
        # Ensure proper spacing between words
        if text_buffer and not text_buffer.endswith(' '):
            text_buffer += ' '
        text_buffer += new_text + " "
        last_buffer_update = time.time()
        debug_timer.mark("buffer_updated", f"Buffer updated: {len(text_buffer)} chars")
    
    # Process buffer for word-based chunks
    await process_text_buffer(tts)
    
    # Update tracking
    tts_sent_length = len(full_text)
    last_tts_text = full_text

async def process_text_buffer(tts):
    """Process text buffer using hybrid word-based + punctuation chunking"""
    global text_buffer, last_buffer_update, last_tts_time, tts_queue, tts_processing
    
    if not text_buffer.strip():
        return
    
    # Split text into words
    words = text_buffer.strip().split()
    
    # Create chunks with punctuation awareness
    chunks = []
    current_chunk_words = []
    
    for i, word in enumerate(words):
        current_chunk_words.append(word)
        
        # Check if we should break the chunk
        should_break = False
        
        # Break if punctuation is found and we have at least 2 words (HIGHER PRIORITY)
        if ENABLE_PUNCTUATION_BREAKS and len(current_chunk_words) >= 2:
            for punct in PUNCTUATION_CHARS:
                if word.endswith(punct):
                    should_break = True
                    break
        
        # Break if we reached the word limit (LOWER PRIORITY)
        if not should_break and len(current_chunk_words) >= WORDS_PER_CHUNK:
            should_break = True
        
        # Create chunk if we should break
        if should_break:
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            current_chunk_words = []
    
    # Add complete chunks to queue
    for chunk in chunks:
        if chunk and len(chunk.split()) >= 2:  # At least 2 words per chunk
            tts_queue.append(chunk)
            debug_timer.mark("tts_queued", f"Queued hybrid chunk ({len(chunk.split())} words): {chunk[:50]}...")
    
    # Update buffer to contain remaining words (incomplete chunk)
    text_buffer = " ".join(current_chunk_words) + " " if current_chunk_words else ""
    
    # Timeout fallback: if buffer has been waiting too long, add it to queue
    if text_buffer.strip() and (time.time() - last_buffer_update) > TIMEOUT_SECONDS:
        remaining_text = text_buffer.strip()
        if len(remaining_text.split()) >= MIN_WORDS_FOR_TIMEOUT:
            tts_queue.append(remaining_text)
            debug_timer.mark("tts_queued", f"Queued timeout chunk ({len(remaining_text.split())} words): {remaining_text[:50]}...")
            text_buffer = ""  # Clear buffer after queuing
    
    # Process queue if not already processing
    if tts_queue and not tts_processing:
        await process_tts_queue(tts)

async def process_tts_queue(tts):
    """Process TTS queue sequentially"""
    global tts_queue, tts_processing, last_tts_time
    
    if tts_processing:
        return
    
    tts_processing = True
    
    try:
        while tts_queue:
            # Update activity timestamp
            update_activity()
            
            chunk = tts_queue.pop(0)  # Get first chunk from queue
            debug_timer.mark("tts_sending", f"Sending queued chunk ({len(chunk.split())} words): {chunk[:50]}...")
            await tts.speak(chunk)
            last_tts_time = time.time()
            debug_timer.mark("tts_sent", f"Queued chunk sent to TTS ({len(chunk)} chars)")
            
            # Reduced delay between chunks for faster processing
            await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05 seconds
    
    finally:
        tts_processing = False

async def mic_stream_generator() -> AsyncGenerator[bytes, None]:
    """Generate audio chunks from microphone with silence detection"""
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
        
        silent_chunks = 0
        last_speech_time = time.time()
        
        while True:
            try:
                audio_data = stream.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
                
                # Check audio level for silence detection
                import struct
                audio_samples = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
                audio_level = max(abs(sample) for sample in audio_samples) / 32768.0
                
                if audio_level < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= SILENCE_CHUNKS:
                        # Extended silence detected - wait for remaining processing
                        print(f"\n🔇 Silence detected for {SILENCE_DURATION_MS/1000:.1f}s - waiting for remaining text...")
                        
                        # Wait additional time for Speechmatics to process remaining audio
                        # This accounts for max_delay (0.7s) + buffer processing time
                        await asyncio.sleep(1.0)  # Reduced from 2.0 to 1.0 seconds for faster response
                        
                        print(f"🔇 Audio input stopped after processing delay")
                        break
                else:
                    silent_chunks = 0
                    last_speech_time = time.time()
                
                if len(audio_data) % BYTES_PER_SAMPLE != 0:
                    padding_needed = BYTES_PER_SAMPLE - (len(audio_data) % BYTES_PER_SAMPLE)
                    audio_data += b'\x00' * padding_needed
                
                yield audio_data
                await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)  # More precise timing
                
            except Exception as e:
                print(f"❌ STT: Microphone error: {e}")
                break
                
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

async def speechmatics_handler(websocket, tts=None):
    """Handle Speechmatics messages"""
    global final_transcript, current_partial
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("message")
                
                if message_type == "RecognitionStarted":
                    session_id = data.get("id", "unknown")
                    debug_timer.mark("stt_recognition_started", f"STT recognition started (ID: {session_id})")
                    
                elif message_type == "AddTranscript":
                    transcript_segment = data.get("metadata", {}).get("transcript", "")
                    final_transcript += transcript_segment
                    current_partial = ""
                    debug_timer.mark("stt_final_transcript", f"Final transcript: {transcript_segment[:50]}...")
                    print_transcript()
                    
                    # Update activity timestamp
                    update_activity()
                    
                    # Send to TTS if we have accumulated substantial text
                    if tts:
                        # Get the full accumulated text since last TTS
                        full_text = final_transcript + current_partial
                        if len(full_text.strip().split()) >= WORDS_PER_CHUNK:  # At least one complete chunk
                            await send_to_tts_if_new(tts, full_text)
                    
                elif message_type == "AddPartialTranscript":
                    full_partial = data.get("metadata", {}).get("transcript", "")
                    
                    if full_partial.startswith(final_transcript):
                        new_partial = full_partial[len(final_transcript):]
                        if new_partial != current_partial:
                            current_partial = new_partial
                            debug_timer.mark("stt_partial_transcript", f"Partial transcript: {new_partial[:50]}...")
                            print_transcript()
                    else:
                        if full_partial != current_partial:
                            current_partial = full_partial
                            debug_timer.mark("stt_partial_transcript", f"Partial transcript: {full_partial[:50]}...")
                            print_transcript()
                    
                elif message_type == "EndOfTranscript":
                    print("\n🎤 STT: End of transcript")
                    break
                    
                elif message_type == "Error":
                    error_type = data.get("type", "unknown")
                    error_reason = data.get("reason", "No reason provided")
                    print(f"\n❌ STT: Error - {error_type}: {error_reason}")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ STT: Failed to parse message: {e}")
                break
            except Exception as e:
                print(f"\n❌ STT: Message handling error: {e}")
                break
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ STT: Connection closed: {e}")
        raise  # Re-raise to trigger reconnection
    except Exception as e:
        print(f"\n❌ STT: Unexpected error: {e}")
        raise  # Re-raise to trigger reconnection

async def force_flush_remaining_text(tts):
    """Force flush any remaining text immediately"""
    global text_buffer, tts_queue
    
    if text_buffer.strip():
        remaining_text = text_buffer.strip()
        if len(remaining_text.split()) >= 1:  # Even single words
            tts_queue.append(remaining_text)
            debug_timer.mark("tts_force_flush", f"Force flushing remaining text: {remaining_text[:50]}...")
            text_buffer = ""
    
    if tts_queue:
        await process_tts_queue(tts)
        debug_timer.mark("tts_force_flushed", "Queue force flushed")

async def speechmatics_sender(websocket):
    """Send audio to Speechmatics"""
    global audio_chunks_sent
    
    # Configure audio settings
    audio_settings = AudioFormat(
        encoding="pcm_s16le",
        sample_rate=RATE
    )
    
    # Configure transcription settings - Optimized for speed and responsiveness
    transcription_config = TranscriptionConfig(
        language="ru",
        max_delay=0.7,  # Minimum allowed by Speechmatics API
        max_delay_mode="flexible",
        operating_point="enhanced",
        enable_partials=True  # Critical for low latency!
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
    
    try:
        await websocket.send(json.dumps(start_message))
        print("🎤 STT: Recognition started")
        
        # Send audio chunks
        audio_generator = mic_stream_generator()
        async for audio_chunk in audio_generator:
            try:
                await websocket.send(audio_chunk)
                audio_chunks_sent += 1
            except websockets.exceptions.ConnectionClosed:
                print("❌ STT: Connection closed while sending audio")
                raise  # Re-raise to trigger reconnection
            except Exception as e:
                print(f"❌ STT: Failed to send audio: {e}")
                raise  # Re-raise to trigger reconnection
        
        # End stream
        end_message = {
            "message": "EndOfStream",
            "last_seq_no": audio_chunks_sent
        }
        await websocket.send(json.dumps(end_message))
        print(f"\n🎤 STT: End of stream sent ({audio_chunks_sent} chunks)")
        
        # Force flush any remaining text immediately
        # Note: TTS instance will be passed from the calling function
        debug_timer.mark("audio_stream_ended", "Audio stream ended, ready for final processing")
        
    except Exception as e:
        print(f"❌ STT: Sender error: {e}")
        raise  # Re-raise to trigger reconnection

async def tts_monitor_task(tts):
    """Monitor TTS system"""
    try:
        debug_timer.mark("tts_monitor_started", "TTS monitor started")
        
        # Keep the task alive to handle TTS requests
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"❌ TTS: Monitor failed: {e}")

async def flush_pending_tts(tts):
    """Flush any remaining text in buffer and queue"""
    global text_buffer, tts_queue
    
    # Add any remaining buffer text to queue
    if text_buffer.strip():
        remaining_text = text_buffer.strip()
        if len(remaining_text.split()) >= 2:  # At least 2 words for flush
            tts_queue.append(remaining_text)
            debug_timer.mark("tts_flush", f"Flushing buffer to queue: {remaining_text[:50]}...")
            text_buffer = ""  # Clear buffer
    
    # Process any remaining queue items
    if tts_queue:
        await process_tts_queue(tts)
        debug_timer.mark("tts_flushed", "Queue flushed")

async def health_monitor_task():
    """Monitor system health and prevent stuck states"""
    global last_activity_time, session_start_time, system_healthy, tts_processing
    
    print("🏥 Health monitor started")
    
    while system_healthy:
        try:
            current_time = time.time()
            
            # Check session duration
            session_duration = current_time - session_start_time
            if session_duration > MAX_SESSION_DURATION:
                print(f"⚠️ Session timeout after {session_duration:.1f}s - forcing restart")
                system_healthy = False
                break
            
            # Check for activity timeout
            time_since_activity = current_time - last_activity_time
            if time_since_activity > CONNECTION_HEALTH_TIMEOUT:
                print(f"⚠️ No activity for {time_since_activity:.1f}s - forcing restart")
                system_healthy = False
                break
            
            # Check TTS queue processing
            if tts_processing and time_since_activity > TTS_QUEUE_TIMEOUT:
                print(f"⚠️ TTS queue stuck for {time_since_activity:.1f}s - forcing reset")
                tts_processing = False  # Force reset TTS processing
            
            # Log health status every 30 seconds
            if int(current_time) % 30 == 0:
                print(f"🏥 Health check: Session {session_duration:.1f}s, Last activity {time_since_activity:.1f}s ago")
            
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
        except Exception as e:
            print(f"❌ Health monitor error: {e}")
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    
    print("🏥 Health monitor stopped")

def update_activity():
    """Update last activity timestamp"""
    global last_activity_time
    last_activity_time = time.time()

async def run_single_connection_session(tts):
    """Run a single STT connection session with TTS integration"""
    global final_transcript, current_partial, audio_chunks_sent
    
    # Reset session state
    final_transcript = ""
    current_partial = ""
    audio_chunks_sent = 0
    
    debug_timer.mark("stt_connecting", "Connecting to Speechmatics")
    
    async with websockets.connect(
        SPEECHMATICS_URL,
        additional_headers={"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"},
        ping_interval=20,      # Send ping every 20 seconds
        ping_timeout=60        # Wait 60 seconds for pong
    ) as websocket:
        debug_timer.mark("stt_connected", "STT connected to Speechmatics")
        
        # Run STT tasks
        receiver_task = asyncio.create_task(speechmatics_handler(websocket, tts))
        sender_task = asyncio.create_task(speechmatics_sender(websocket))
        
        # Wait for all tasks
        done, pending = await asyncio.wait(
            [receiver_task, sender_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cleanup
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Force flush any remaining text after audio ends
        await force_flush_remaining_text(tts)
        
        # Additional delay to ensure all processing is complete
        print("🔄 Waiting for final processing to complete...")
        await asyncio.sleep(1.0)  # Wait 1 second for any final processing

async def run_with_reconnection(tts):
    """Run the system with automatic reconnection and exponential backoff"""
    reconnect_delay = INITIAL_RECONNECT_DELAY
    session_count = 0
    
    print("🔄 Starting robust single connection with automatic reconnection")
    
    while True:
        try:
            session_count += 1
            print(f"\n🔄 Starting session #{session_count}")
            
            # Run a single connection session
            await run_single_connection_session(tts)
            
            # If we get here, the session ended gracefully
            print(f"✅ Session #{session_count} completed gracefully")
            break
            
        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ Session #{session_count} failed: Connection closed - {e}")
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"❌ Session #{session_count} failed: Invalid status code {e.status_code}")
        except Exception as e:
            print(f"❌ Session #{session_count} failed: {e}")
        
        # Calculate reconnection delay with exponential backoff and jitter
        if RECONNECT_JITTER:
            jitter = random.uniform(0.5, 1.5)
            sleep_duration = min(reconnect_delay * jitter, MAX_RECONNECT_DELAY)
        else:
            sleep_duration = min(reconnect_delay, MAX_RECONNECT_DELAY)
        
        print(f"🔄 Reconnecting in {sleep_duration:.2f} seconds...")
        await asyncio.sleep(sleep_duration)
        
        # Exponential backoff
        reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)

async def main():
    """Main function"""
    print("🚀 Starting Combined STT + TTS System - Robust Single Connection")
    print(f"📋 STT: Real-time transcription with keepalives")
    print(f"🔊 TTS: Parallel audio output")
    print(f"📝 Chunking: {WORDS_PER_CHUNK} words per chunk")
    if ENABLE_PUNCTUATION_BREAKS:
        print(f"🔤 Punctuation breaks: {PUNCTUATION_CHARS}")
    else:
        print(f"🔤 Punctuation breaks: disabled")
    print(f"⏱️  Timeout: {TIMEOUT_SECONDS}s (min {MIN_WORDS_FOR_TIMEOUT} words)")
    print(f"🔇 Silence detection: {SILENCE_DURATION_MS/1000:.1f}s threshold")
    print(f"🔄 Reconnection: exponential backoff (initial: {INITIAL_RECONNECT_DELAY}s, max: {MAX_RECONNECT_DELAY}s)")
    print(f"🏥 Health monitoring: {HEALTH_CHECK_INTERVAL}s intervals, {CONNECTION_HEALTH_TIMEOUT}s activity timeout")
    
    # Create TTS audio player
    tts_player = TTSAudioPlayer()
    tts_player.start()
    
    # Create TTS client
    tts = ElevenLabsTTS(tts_player)
    
    try:
        # Connect TTS
        await tts.connect()
        
        # Start TTS monitor task
        tts_task = asyncio.create_task(tts_monitor_task(tts))
        
        # Start health monitor task
        health_task = asyncio.create_task(health_monitor_task())
        
        # Run with reconnection
        await run_with_reconnection(tts)
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ STT: Connection failed (Status: {e.status_code})")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"❌ STT: Connection closed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Stop health monitoring
        system_healthy = False
        
        # Cleanup
        await flush_pending_tts(tts)
        await tts.close()
        tts_player.stop()
        
        # Cancel health task
        if health_task:
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
        
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